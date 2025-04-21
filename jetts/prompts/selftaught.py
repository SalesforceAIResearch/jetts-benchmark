from ..data import Query, Response
from jetts.prompts import get_partial_response

single_instance_rate_max_score = 5

PROMPT_PAIRWISE_SYSTEM="""
Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user question displayed below. You should choose the assistant that follows the user\'s instructions and answers the user\'s question better. Your evaluation should consider factors such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of their responses. Begin your evaluation by comparing the two responses and provide a short explanation. Avoid any position biases and ensure that the order in which the responses were presented does not influence your decision. Do not allow the length of the responses to influence your evaluation. Do not favor certain names of the assistants. Be as objective as possible. {partial_response_note}After providing your explanation, output your final verdict by strictly following this format: \\"[[A]]\\" if assistant A is better, \\"[[B]]\\" if assistant B is better.'
""".strip()

PROMPT_PAIRWISE_COMPARISON="""
[User Question]
{query_text}

[The Start of Assistant A's Answer]
{response_A}
[The End of Assistant A's Answer]

[The Start of Assistant B's Answer]
{response_B}
[The End of Assistant B's Answer]
""".strip()

def parse_pairwise_judgment(judge_output, return_critique):
    if return_critique:
        raise Exception('This judge cannot generate critiques')
    if '[[A]]' in judge_output:
        return 0
    elif '[[B]]' in judge_output:
        return 1
    else:
        return None


def render_pairwise_prompt(query: Query, response0: Response, response1: Response, partial_response=False):
    #assert not partial_response, 'Partial response template has not been implemented'
    system_prompt = PROMPT_PAIRWISE_SYSTEM.format(partial_response_note=get_partial_response(partial_response, prefix='', suffix=' '))
    prompt = PROMPT_PAIRWISE_COMPARISON.format(query_text=query.content, response_A=response0.content, response_B=response1.content)
    return [dict(role='system', content=system_prompt), 
            dict(role='user', content=prompt)]
