
import time
from openai import OpenAI
from datasets import Dataset
from datasets.utils import logging

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
    def __init__(self, ip=None, port=None, base_url=None, api_key='sample-api-key', model_name=None, batch_size=None, 
                 output_field_name='output', output_error_val='Error', max_connection_tries=None, delay_time=None, temperature=0, max_tokens=1024, top_p=1.0):
        if base_url is not None:
            assert ip == port == None, 'If base_url is specified, ip and port should be None.'
            self.base_url = base_url
        else:
            assert (ip is not None) and (port is not None), 'If base_url is not specified, ip and port should be specified.'
            self.base_url = f'http://{ip}:{port}/v1'
        wait_until_server_ready(self.base_url, api_key=api_key, max_connection_tries=max_connection_tries, delay_time=delay_time)
        self.api_key = api_key
        if batch_size is None:
            batch_size = 50
        self.batch_size = batch_size
        self.model_name = self.get_model_name(model_name)
        self.output_field_name = output_field_name
        self.output_error_val = output_error_val
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
    
    def get_model_name(self, model_name):
        if model_name is not None:
            return model_name
        client = OpenAI(base_url=self.base_url, api_key=self.api_key)
        model_list = list(client.models.list())
        assert len(model_list) == 1, 'The vllm server is serving multiple models? This doesn\'t seem right...'
        return model_list[0].id
    
    def chat_map(self, instance):
        try:
            conversation = instance['conversation']
            client = OpenAI(base_url=self.base_url, api_key=self.api_key)
            response = client.chat.completions.create(
                model=self.model_name, 
                messages=conversation, temperature=self.temperature, max_tokens=self.max_tokens, top_p=self.top_p)
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
