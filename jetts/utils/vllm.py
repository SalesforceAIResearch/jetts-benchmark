
import time
from openai import OpenAI
from datasets import Dataset
from datasets.utils import logging
from msgspec.structs import asdict
from transformers import AutoTokenizer as AT
from vllm import SamplingParams
from vllm.inputs import TokensPrompt

MAX_CONNECTION_TRIES = 120 # number of times to try connecting judge and refiner endpoints
DELAY_TIME = 10 # sleep for 10s; 120 x 10 = 20 minutes before giving up

def wait_until_server_ready(base_url, api_key='sample-api-key', max_connection_tries=None, delay_time=None):
    if 'together' in base_url:
        return
    if max_connection_tries is None:
        max_connection_tries = MAX_CONNECTION_TRIES
    if delay_time is None:
        delay_time = DELAY_TIME
    for i in range(max_connection_tries):
        try:
            client = OpenAI(base_url=base_url, api_key=api_key)
            client.models.list()
            return
        except:
            print(f'Waiting for server to be ready... {i+1} / {max_connection_tries}')
            time.sleep(delay_time)
    raise ConnectionError(f'Cannot connect to server after {max_connection_tries} tries.')

class VllmEndpointCaller():
    def __init__(self, sampling_params, ip=None, port=None, base_url=None, api_key='sample-api-key', model_name=None, batch_size=None, 
                 output_field_name='output', output_error_val='Error', max_connection_tries=None, delay_time=None):
        if base_url is not None:
            assert ip == port == None, 'If base_url is specified, ip and port should be None.'
            self.base_url = base_url
        else:
            assert (ip is not None) and (port is not None), 'If base_url is not specified, ip and port should be specified.'
            self.base_url = f'http://{ip}:{port}/v1'
        wait_until_server_ready(self.base_url, api_key=api_key, max_connection_tries=max_connection_tries, delay_time=delay_time)
        assert isinstance(sampling_params, SamplingParams)
        self.sampling_params = sampling_params
        self.api_key = api_key
        if batch_size is None:
            batch_size = 50
        self.batch_size = batch_size
        self.model_name = self.get_model_name(model_name)
        self.output_field_name = output_field_name
        self.output_error_val = output_error_val
        self.tokenizer = AT.from_pretrained(self.model_name)
    
    def get_model_name(self, model_name):
        if model_name is not None:
            return model_name
        client = OpenAI(base_url=self.base_url, api_key=self.api_key)
        model_list = list(client.models.list())
        assert len(model_list) == 1, 'The vllm server is serving multiple models? This doesn\'t seem right...'
        return model_list[0].id
    
    def chat_map(self, instance):
        try:
            sp = self.sampling_params
            conversation = instance['conversation']
            client = OpenAI(base_url=self.base_url, api_key=self.api_key)
            response = client.chat.completions.create(
                model=self.model_name, 
                messages=conversation, temperature=sp.temperature, max_tokens=sp.max_tokens, top_p=sp.top_p)
            instance[self.output_field_name] = response.choices[0].message.content
        except Exception as e:
            print(f'Error on instance: {instance}\n\nError: {e}')
            instance[self.output_field_name] = self.output_error_val
        return instance
    
    def generate_batch(self, conversations, use_tqdm=True):
        if len(conversations) == 0:
            return []
        cur_use_tqdm = logging.is_progress_bar_enabled()
        dataset_data = [{'conversation': conv} for conv in conversations]
        dataset = Dataset.from_list(dataset_data)
        if use_tqdm:
            logging.enable_progress_bar()
        else:
            logging.disable_progress_bar()
        dataset = dataset.map(self.chat_map, num_proc=min(len(conversations), self.batch_size))
        if cur_use_tqdm:
            logging.enable_progress_bar()
        else:
            logging.disable_progress_bar()
        return dataset[self.output_field_name]
    
    def complete_map(self, instance):
        try:
            if 'sampling_params' in instance:
                sp = SamplingParams(**instance['sampling_params'])
            else:
                sp = self.sampling_params
            prompt = instance['prompt']
            # type check that prompt is a list of integers
            assert isinstance(prompt, list) and all(isinstance(i, int) for i in prompt), 'Prompt must be a list of integers.'
            extra_body = instance['extra_body']
            return_metadata = instance['return_metadata']
            client = OpenAI(base_url=self.base_url, api_key=self.api_key)
            response = client.completions.create(
                model=self.model_name, 
                prompt=prompt, temperature=sp.temperature, max_tokens=sp.max_tokens, top_p=sp.top_p, logprobs=True, n=instance['num'], 
                extra_body=extra_body)
            tokens_lst = []
            metadata_lst = []
            for choice in response.choices:
                str_tokens = choice.logprobs.tokens
                text = ''.join(str_tokens)
                tokens = self.tokenizer(text, add_special_tokens=False)['input_ids']
                if str_tokens[-1] == '':
                    str_tokens.pop()
                if self.tokenizer.decode(tokens) != ''.join(str_tokens):
                    print(choice.logprobs.tokens)
                    raise Exception('retokenization error')
                tokens_lst.append(tokens)
                metadata_lst.append(dict(text=choice.text, 
                                         finish_reason=choice.finish_reason, 
                                         stop_reason=(choice.stop_reason if choice.stop_reason is not None else -1)))
            if instance['num'] == 1 and instance['squeeze']:
                tokens_lst = tokens_lst[0]
                metadata_lst = metadata_lst[0]
            instance[self.output_field_name] = tokens_lst
            if return_metadata:
                instance['metadata'] = metadata_lst
        except Exception as e:
            print(f'Error on instance: {instance}\n\nError: {e}')
            instance[self.output_field_name] = self.output_error_val
        return instance
    
    def complete_batch(self, prompts, stop_token_ids=None, use_tqdm=True, return_metadata=False, sampling_params_lst: list[SamplingParams]=None, n=1, squeeze=True):
        if len(prompts) == 0:
            return []
        if stop_token_ids is None:
            stop_token_ids = []
        if isinstance(n, int):
            n = [n] * len(prompts)
        extra_body = {'stop_token_ids': stop_token_ids}
        cur_use_tqdm = logging.is_progress_bar_enabled()
        if sampling_params_lst is None:
            dataset_data = [{'prompt': prompt, 'extra_body': extra_body, 'return_metadata': return_metadata, 'num': ne, 'squeeze': squeeze} 
                            for prompt, ne in zip(prompts, n)]
        else:
            if isinstance(sampling_params_lst, SamplingParams):
                sampling_params_lst = [sampling_params_lst] * len(prompts)
            sampling_params_lst = [asdict(sp) if sp is not None else None for sp in sampling_params_lst]
            # the output_kind (default being RequestOutputKind.CUMULATIVE) can't be serialized by pyarrow
            [sp.pop('output_kind', None) for sp in sampling_params_lst]
            dataset_data = [{'prompt': prompt, 'extra_body': extra_body, 'return_metadata': return_metadata, 'num': ne, 'squeeze': squeeze, 
                             'sampling_params': sp} for prompt, ne, sp in zip(prompts, n, sampling_params_lst)]
        dataset = Dataset.from_list(dataset_data)
        if use_tqdm:
            logging.enable_progress_bar()
        else:
            logging.disable_progress_bar()
        dataset = dataset.map(self.complete_map, num_proc=min(len(prompts), self.batch_size))
        if cur_use_tqdm:
            logging.enable_progress_bar()
        else:
            logging.disable_progress_bar()
        if not return_metadata:
            return dataset[self.output_field_name]
        else:
            if n == 1 and squeeze:
                return list(zip(dataset[self.output_field_name], dataset['metadata']))
            else:
                return list(zip(dataset[self.output_field_name], dataset['metadata']))
    
class VllmCaller():
    def __init__(self, llm, unpack_hack=False):
        self.llm = llm
        self.eos_tokens = llm.llm_engine.generation_config_fields['eos_token_id']
        if isinstance(self.eos_tokens, int):
            self.eos_tokens = [self.eos_tokens]
        self.unpack_hack = unpack_hack

    def rewrite_stop_reason(self, s):
        if s in self.eos_tokens:
            print(f'rewriting {s} to None')
            return None
        else:
            return s

    def __call__(self, tks_lst, sampling_params):
        if not self.unpack_hack:
            inputs = [TokensPrompt(prompt_token_ids=tks) for tks in tks_lst]
            result = self.llm.generate(inputs, sampling_params)
            return ([[o.token_ids for o in res.outputs] for res in result], 
                    [[o.finish_reason for o in res.outputs] for res in result], 
                    [[self.rewrite_stop_reason(o.stop_reason) for o in res.outputs] for res in result])
        else:
            assert isinstance(sampling_params, list) and len(tks_lst) == len(sampling_params)
            unpacked_tks_lst = []
            unpacked_sampling_params_lst = []
            for tks, sp in zip(tks_lst, sampling_params):
                n = sp.n
                sp_clone = sp.clone()
                sp_clone.n = 1
                unpacked_tks_lst.extend([tks] * n)
                unpacked_sampling_params_lst.extend([sp_clone] * n)
            unpacked_inputs = [TokensPrompt(prompt_token_ids=tks) for tks in unpacked_tks_lst]
            result = self.llm.generate(unpacked_inputs, unpacked_sampling_params_lst)
            packed_token_ids_lst = []
            packed_finish_reason_lst = []
            packed_stop_reason_lst = []
            idx = 0
            for sp in sampling_params:
                n = sp.n
                packed_token_ids_lst.append([])
                packed_finish_reason_lst.append([])
                packed_stop_reason_lst.append([])
                for _ in range(n):
                    output_lst = result[idx].outputs
                    assert len(output_lst) == 1
                    o = output_lst[0]
                    packed_token_ids_lst[-1].append(o.token_ids)
                    packed_finish_reason_lst[-1].append(o.finish_reason)
                    packed_stop_reason_lst[-1].append(self.rewrite_stop_reason(o.stop_reason))
                    idx += 1
            assert idx == len(result)
            return (packed_token_ids_lst, packed_finish_reason_lst, packed_stop_reason_lst)
