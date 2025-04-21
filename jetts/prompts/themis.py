
from jetts.prompts import get_partial_response
from ..data import Query, Response
import math

single_instance_rate_max_score = 5

PROMPT_SINGLE_RATING = """###Instruction###

Please act as an impartial and helpful evaluator for natural language generation (NLG), and the audience is an expert in the field.
Your task is to evaluate the quality of responses for a given user instruction strictly based on the given evaluation criterion.
Begin the evaluation by providing your analysis concisely and accurately, and then on the next line, start with \"Rating:\" followed by your rating on a Likert scale from 1 to 5 (higher means better).
You MUST keep to the strict boundaries of the evaluation criterion and focus solely on the issues and errors involved; otherwise, you will be penalized.
Make sure you read and understand these instructions, as well as the following evaluation criterion and example content, carefully.

###Evaluation Criterion###
(1) You should prioritize evaluating whether the output honestly/precisely/closely executes the instruction, then consider its helpfulness, accuracy, level of detail, harmlessness, etc. 
(2) Responses should NOT contain more/less than what the instruction asks for, as such responses do NOT precisely execute the instruction.
(3) You should avoid any potential bias and your judgment should be as objective as possible. Here is a potential source of bias:
- The length of the response should NOT affect your judgement, as a longer response does not necessarily correspond to a better response. When making your decision, evaluate if the response length is appropriate for the given instruction.{partial_response_note}

###Example###
Instruction:
{query_text}

Response:
{response}

###Your Evaluation###
"""

PROMPT_SINGLE_RATING_ADDITIVE = """###Instruction###
Please act as an impartial and helpful evaluator for natural language generation (NLG), and the audience is an expert in the field.
Your task is to evaluate the quality of responses for a given user instruction strictly based on the given evaluation criterion.
Begin the evaluation by providing your analysis concisely and accurately, and then on the next line, start with \"Rating:\" followed by your rating on a Likert scale from 1 to 5 (higher means better).
You MUST keep to the strict boundaries of the evaluation criterion and focus solely on the issues and errors involved; otherwise, you will be penalized.
Make sure you read and understand these instructions, as well as the following evaluation criterion and example content, carefully.

###Evaluation Criterion###
- Add one point if the response is relevant and provides some information related to the user's inquiry, even if it is incomplete or contains some irrelevant content.
- Add a second point if the response addresses a substantial portion of the user's question, but does not completely resolve the query or provide a direct answer.
- Add a third point if the response answers the basic elements of the user's question in a useful way, regardless of whether it seems to have been written by an AI Assistant or if it has elements typically found in blogs or search results.
- Add a fourth point if the response is clearly written from an AI Assistant's perspective, addressing the user's question directly and comprehensively, and is well-organized and helpful, even if there is slight room for improvement in clarity, conciseness or focus.
- Add a fifth point for a response that is impeccably tailored to the user's question by an AI Assistant, without extraneous information, reflecting expert knowledge, and demonstrating a high-quality, engaging, and insightful answer.{partial_response_note}

###Example###
Instruction:
{query_text}

Response:
{response}

###Your Evaluation###
"""


def parse_single_instance_rate_judgment(judge_output, return_critique):
    judgement = judge_output.split('\n')[-1].strip()
    critique = '\n'.join(judge_output.split('\n')[:-1]).strip()
    if judgement.startswith("Rating:"):
        try: 
            rating = int(judgement.split('Rating:')[-1].strip())
            if math.isfinite(rating):
                if return_critique:
                    return rating, critique
                else:
                    return rating
        except:
            if return_critique:
                return None, None
            else:
                return None
    else:
        if return_critique:
            return None, None
        else:
            return None

def render_single_instance_rate_prompt(query: Query, response: Response, which='single-likert', partial_response=False):
    assert which in ['single-likert', 'single-additive']
    if 'single-likert' in which:
        prompt = PROMPT_SINGLE_RATING
    elif 'single-additive' in which:
        prompt = PROMPT_SINGLE_RATING_ADDITIVE
    prompt = prompt.format(query_text=query.content, response=response.content, partial_response_note=get_partial_response(partial_response))
    return [dict(role='user', content=prompt)]
