import importlib

PARTIAL_RESPONSE_NOTE='Important note: the model response is generated with streaming, meaning that it may not be finished yet but we still want to get a sense of the partial progress so far. If this is the case, you should provide judgment and feedback on the currently generated part, focusing on its correctness and promise of leading to a correct final solution.'

def get_partial_response(partial_response, prefix='\n\n', suffix=''):
    if partial_response:
        return prefix + PARTIAL_RESPONSE_NOTE + suffix
    else:
        return ''

judge_module_map = {
    'prometheus-eval/prometheus-7b-v2.0': 'prometheus',
    'Skywork/Skywork-Critic-Llama-3.1-8B': 'skywork',
    'NCSOFT/Llama-3-OffsetBias-8B': 'offsetbias',
    'PKU-ONELab/Themis': 'themis',
    'Skywork/Skywork-Critic-Llama-3.1-70B': 'skywork',
    'facebook/Self-taught-evaluator-llama3.1-70B': 'selftaught',
    'prometheus-eval/prometheus-8x7b-v2.0': 'prometheus',
    'meta-llama/Llama-3.1-8B-Instruct': 'sfrjudge',
}

def load_prompter(model):
    if model not in judge_module_map:
        raise Exception(f'Judge {model} is not implemented!')
    module_name = judge_module_map[model]
    prompter = importlib.import_module('jetts.prompts.{}'.format(module_name))
    return prompter
