from ..data import Query, Response
from jetts.prompts import get_partial_response

single_instance_rate_max_score = 5

PROMPT_PAIRWISE_COMPARISON="""
###Task Description:
An instruction (might include an Input inside it) and two responses to evaluate are given.
1. Write a detailed feedback that assess the quality of the responses based on whether the output honestly/precisely/closely executes the instruction, then consider its helpfulness, accuracy, level of detail, harmlessness, etc.
2. After writing a feedback, choose a better response between Response A and Response B.
3. The output format should look as follows: "Feedback: (write a feedback for criteria) [RESULT] (A or B)"
4. Please do not generate any other opening, closing, and explanations.{partial_response_note}

###Instruction:
{query_text}

###Response A:
{response_A}

###Response B:
{response_B}

###Feedback: 
""".strip()


def parse_pairwise_judgment(judge_output, return_critique):
    critique_judgement = judge_output.split('[RESULT]')
    critique = critique_judgement[0].strip()
    judgement = critique_judgement[-1].strip()
    if return_critique:
        if judgement == 'A':
            return 0, critique
        elif judgement == 'B':
            return 1, critique
        else:
            return None, None
    else:
        if judgement == 'A':
            return 0
        elif judgement == 'B':
            return 1
        else:
            return None


PROMPT_SINGLE_RATING="""
###Task Description:
An instruction (might include an Input inside it) and a response to evaluate are given.
1. Write a detailed feedback that assess the quality of the response based on whether the output honestly/precisely/closely executes the instruction, then consider its helpfulness, accuracy, level of detail, harmlessness, etc.
2. After writing a feedback, write a score that is an integer between 1 and 5.
3. The output format should look as follows: \"Feedback: (write a feedback for criteria) [RESULT] (an integer number between 1 and 5)\"
4. Please do not generate any other opening, closing, and explanations.{partial_response_note}

###The instruction to evaluate:
{query_text}

###Response to evaluate:
{response}

###Feedback: 
""".strip()

PROMPT_SINGLE_RATING_ADDITIVE="""
###Task Description:
An instruction (might include an Input inside it), a response to evaluate, and a score rubric representing a evaluation criteria are given.
1. Write a detailed feedback that assess the quality of the response strictly based on the given score rubric, not evaluating in general.
2. After writing a feedback, write a score that is an integer between 1 and 5. You should refer to the score rubric.
3. The output format should look as follows: "(write a feedback for criteria) [RESULT] (an integer number between 1 and 5)"
4. Please do not generate any other opening, closing, and explanations.{partial_response_note}

###The instruction to evaluate:
{query_text}

###Response to evaluate:
{response}

###Score Rubrics:
- Add one point if the response is relevant and provides some information related to the user's inquiry, even if it is incomplete or contains some irrelevant content.
- Add a second point if the response addresses a substantial portion of the user's question, but does not completely resolve the query or provide a direct answer.
- Add a third point if the response answers the basic elements of the user's question in a useful way, regardless of whether it seems to have been written by an AI Assistant or if it has elements typically found in blogs or search results.
- Add a fourth point if the response is clearly written from an AI Assistant's perspective, addressing the user's question directly and comprehensively, and is well-organized and helpful, even if there is slight room for improvement in clarity, conciseness or focus.
- Add a fifth point for a response that is impeccably tailored to the user's question by an AI Assistant, without extraneous information, reflecting expert knowledge, and demonstrating a high-quality, engaging, and insightful answer.
""".strip()

def parse_single_instance_rate_judgment(judge_output, return_critique):
    critique_judgement = judge_output.split('[RESULT]')
    critique = critique_judgement[0].strip()
    judgement = critique_judgement[-1].strip()
    if return_critique:
        if judgement in ['1', '2', '3', '4', '5']:
            return int(judgement), critique
        else:
            return None, None
    else:
        if judgement in ['1', '2', '3', '4', '5']:
            return int(judgement)
        else:
            return None

def render_pairwise_prompt(query: Query, response0: Response, response1: Response, partial_response=False):
    prompt = PROMPT_PAIRWISE_COMPARISON.format(query_text=query.content, response_A=response0.content, response_B=response1.content,
                                              partial_response_note=get_partial_response(partial_response))

    return [dict(role='user', content=prompt)]


def render_single_instance_rate_prompt(query: Query, response: Response, which='single-likert', partial_response=False):
    assert which in ['single-likert', 'single-additive']
    if which == 'single-likert':
        prompt = PROMPT_SINGLE_RATING
    elif which == 'single-additive':
        prompt = PROMPT_SINGLE_RATING_ADDITIVE
    prompt = prompt.format(query_text=query.content, response=response.content, partial_response_note=get_partial_response(partial_response))
    return [dict(role='user', content=prompt)]
