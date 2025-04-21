from ..data import Query, Response
from ..utils.vllm import VllmEndpointCaller

from dataclasses import dataclass
from vllm import LLM, SamplingParams

@dataclass
class SpecialRefinementOutput:
    value: str
    metadata: dict = None

    def __post_init__(self):
        assert self.value in ['NO_REFINEMENT_NEEDED', 'REFINEMENT_FAILURE']
        if self.metadata is None:
            self.metadata = dict()

SINGLE_INSTANCE_PROMPT = '''
You help revise a machine-generated response to a user query. Below, you will be provided with the user query and the machine-generated response. You will also be provided with the output of an evaluator model, which gives a score (max being {max_score}) and an explanation for the score.

You should revise and improve the current response, following the evaluator's recommendation. If the evaluator does not identify any area of improvement, you should output "No revision needed." Otherwise, you should output the revised response surrounded by the <revised_response> and </revised_response> tags. You do not need to output anything else.

<query>
{query}
</query>

<original_response>
{response}
</original_response>

<score>
{score} out of {max_score}.
</score>

<explanation>
{explanation}
</explanation>

Your revision (or "No revision needed."):
'''.strip()

PAIRWISE_PROMPT = '''
You help revise machine-generated responses to a user query. Below, you will be provided with the user query and two machine-generated responses. An evaluator considers the merits and flaws in both responses, and gives a judgment on which response is better. In addition, it gives the reason for judgment, which may discuss various merits and flaws.

You should generate a single response by combining the merits of both responses and fixing their flaws, following the evaluator's recommendation. If the evaluator does not identify any flaws of the chosen response and it cannot be enhanced by any additional merits of the other response either, you should output "No revision needed." Otherwise, you should output the revised response surrounded by the <revised_response> and </revised_response> tags. You do not need to output anything else.

<query>
{query}
</query>

<response_A>
{response_A}
</response_A>

<response_B>
{response_B}
</response_B>

<judgment>
Response {choice} is better
</judgment>

<explanation>
{explanation}
</explanation>

Your revision (or "No revision needed."):
'''.strip()

def parse_refinement_output(raw_refined_output, metadata):
    if 'no revision needed' in raw_refined_output.lower():
        response = SpecialRefinementOutput('NO_REFINEMENT_NEEDED', metadata)
        return True, response
    if raw_refined_output.count('<revised_response>') != 1:
        response = SpecialRefinementOutput('REFINEMENT_FAILURE', metadata)
        return True, response
    raw_refined_output = raw_refined_output.split('<revised_response>')[1]
    if raw_refined_output.count('</revised_response>') != 1:
        response = SpecialRefinementOutput('REFINEMENT_FAILURE', metadata)
        return True, response
    content = raw_refined_output.split('</revised_response>')[0].strip()
    return False, content

class Refiner():

    def single_instance_refine(self, query: Query, response: Response, explanation, score, max_score):
        return self.single_instance_refine_batch([query], [response], [explanation], [score], [max_score])[0]

    def single_instance_refine_batch(self, query_lst: list[Query], response_lst: list[Response], explanation_lst, score_lst, max_score_lst):
        raise NotImplementedError
    
    def pairwise_refine(self, query: Query, response_0: Response, response_1: Response, explanation, choice):
        return self.pairwise_refine_batch([query], [response_0], [response_1], [explanation], [choice])[0]
    
    def pairwise_refine_batch(self, query_lst: list[Query], response_0_lst: list[Response], response_1_lst: list[Response], explanation_lst, choice_lst):
        raise NotImplementedError

class DefaultRefiner(Refiner):
    
    def single_instance_refine_batch(self, query_lst: list[Query], response_lst: list[Response], explanation_lst, score_lst, max_score_lst, use_tqdm=True):
        prompts = [SINGLE_INSTANCE_PROMPT.format(query=query.content, response=response.content, explanation=explanation, score=score, max_score=max_score)
                    for query, response, explanation, score, max_score in zip(
                        query_lst, response_lst, explanation_lst, score_lst, max_score_lst
                    )]
        conversations = [[dict(role='user', content=prompt)] for prompt in prompts]
        output_lst = self.generate_batch(conversations, use_tqdm=use_tqdm)
        return [self.single_instance_refinement_to_response(output, orig_response, explanation, score) 
                for output, orig_response, explanation, score in zip(output_lst, response_lst, explanation_lst, score_lst)]
        
    def single_instance_refinement_to_response(self, raw_refined_output, original_response, judge_explanation, judge_score):
        metadata = {'original_response': original_response.content, 
                    'judge_explanation': judge_explanation, 
                    'judge_score': judge_score, 
                    'raw_refined_output': raw_refined_output}
        is_special, response = parse_refinement_output(raw_refined_output, metadata)
        if is_special:
            return response
        else:
            return Response(original_response.query, content=response, metadata=metadata)
    
    def pairwise_refine_batch(self, query_lst: list[Query], response_0_lst: list[Response], response_1_lst: list[Response], explanation_lst, choice_lst, use_tqdm=True):
        prompts = [PAIRWISE_PROMPT.format(query=query.content, response_A=response_0.content, response_B=response_1.content, explanation=explanation, choice=chr(ord('A') + choice))
                    for query, response_0, response_1, explanation, choice in zip(
                        query_lst, response_0_lst, response_1_lst, explanation_lst, choice_lst
                    )]
        conversations = [[dict(role='user', content=prompt)] for prompt in prompts]
        output_lst = self.generate_batch(conversations, use_tqdm=use_tqdm)
        return [self.pairwise_refinement_to_response(output, orig_response_0, orig_response_1, explanation, choice) 
                for output, orig_response_0, orig_response_1, explanation, choice in zip(output_lst, response_0_lst, response_1_lst, explanation_lst, choice_lst)]
    
    def pairwise_refinement_to_response(self, raw_refined_output, original_response_0, original_response_1, judge_explanation, judge_choice):
        assert judge_choice in [0, 1]
        metadata = {'original_response_0': original_response_0.content, 
                    'original_response_1': original_response_1.content,
                    'judge_explanation': judge_explanation, 
                    'judge_choice': judge_choice, 
                    'raw_refined_output': raw_refined_output}
        is_special, response = parse_refinement_output(raw_refined_output, metadata)
        if is_special:
            return response
        else:
            return Response(original_response_0.query, content=response, metadata=metadata)
    
class VllmRefiner(DefaultRefiner):
    def __init__(self, model, sampling_params):
        super().__init__()
        assert isinstance(model, LLM), 'model needs to be a vllm.LLM instance'
        assert isinstance(sampling_params, SamplingParams)
        self.model = model
        self.sampling_params = sampling_params

    def generate_batch(self, conversations, use_tqdm=True):
        responses = self.model.chat(conversations, self.sampling_params, use_tqdm=use_tqdm)
        return [response.outputs[0].text for response in responses]
    
class VllmEndpointRefiner(DefaultRefiner):

    def __init__(self, sampling_params, ip=None, port=None, base_url=None, api_key='sample-api-key', model_name=None, batch_size=None):
        super().__init__()
        self.caller = VllmEndpointCaller(sampling_params, ip, port, base_url, api_key, model_name, batch_size, 
                                         output_field_name='refinement', output_error_val='Refinement error')

    def generate_batch(self, conversations, use_tqdm=True):
        return self.caller.generate_batch(conversations, use_tqdm=use_tqdm)
