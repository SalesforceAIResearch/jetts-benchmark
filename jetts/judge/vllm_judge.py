from ..judge.prompted_judge import PromptedJudge
from ..prompts import load_prompter
from ..utils.vllm import VllmEndpointCaller

from vllm import LLM, SamplingParams

class VllmJudge(PromptedJudge):

    def __init__(self, model, sampling_params, prompter):
        super().__init__(prompter)
        assert isinstance(model, LLM), 'model needs to be a vllm.LLM instance'
        assert isinstance(sampling_params, SamplingParams)
        self.model = model
        self.sampling_params = sampling_params

    def generate_batch(self, conversations, use_tqdm=True):
        responses = self.model.chat(conversations, self.sampling_params, use_tqdm=use_tqdm)
        return [response.outputs[0].text for response in responses]

class VllmEndpointJudge(PromptedJudge):

    def __init__(self, sampling_params, prompter=None, ip=None, port=None, base_url=None, api_key='sample-api-key', model_name=None, batch_size=None):
        self.caller = VllmEndpointCaller(sampling_params, ip, port, base_url, api_key, model_name, batch_size, 
                                         output_field_name='output', output_error_val='Error')
        if prompter is None:
            prompter = load_prompter(self.caller.model_name)
        super().__init__(prompter)

    def generate_batch(self, conversations, use_tqdm=True):
        return self.caller.generate_batch(conversations, use_tqdm=use_tqdm)

