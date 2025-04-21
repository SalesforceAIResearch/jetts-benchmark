
from collections import Counter
import json, os, argparse

def evaluate_gsm8k_math(fn, force_cleanup=False, dataset=None):
    import numpy as np
    from math_verify.metric import math_metric
    from math_verify.parser import LatexExtractionConfig, ExprExtractionConfig
    if dataset is None:
        if 'gsm8k' in fn:
            dataset = 'gsm8k'
        elif 'math' in fn:
            dataset = 'math'
        else:
            raise ValueError(f'Unknown dataset: {fn}')
    if force_cleanup or ('deepseek-math-7b-instruct' in fn and dataset == 'gsm8k'):
        def cleanup(response):
            response = response.strip()
            if '$\\boxed{The final answer is' in response:
                response = response.replace('$\\boxed{The final answer is', 'The final answer is')
                for s in ['}$.', '}$']:
                    if response.endswith(s):
                        response = response[:-len(s)] + '.'
                        break
                    # If the first appears in the response, find the last occurrence and truncate the response to that point
                    # Example: orig:         ''...\\boxed{The final answer is 58}. This means that 58...'' 
                    # After replace above:   ''... 58}.''
                    # After finding last }.: ''...58.''
                    else:
                        last_ind = response.rfind(s)
                        response = response[:last_ind] + '.'
                        break
            return response
    else:
        cleanup = lambda x: x
    verify_func = math_metric(
        gold_extraction_target=(LatexExtractionConfig(), ExprExtractionConfig()),
        pred_extraction_target=(ExprExtractionConfig(), LatexExtractionConfig()),
        aggregation_function=lambda x: (float(np.average(x)), min(x), max(x)),
        precision=6
    )
    num_total = 0
    num_correct = 0
    for line in open(fn):
        data = json.loads(line)
        if dataset == 'gsm8k':
            gold = data['query']['metadata']['input_correct_responses']
        else:
            gold = [data['query']['metadata']['solution']]
        response = data['responses'][0]['content']
        response = [cleanup(response)]
        try:
            grade, _ = verify_func(gold, response)
            num_total += 1
            num_correct += grade[0]
        except:
            num_total += 1
    return num_correct / num_total

def evaluate_gsm8k(fn):
    return evaluate_gsm8k_math(fn, force_cleanup=False, dataset='gsm8k')

def evaluate_math(fn):
    return evaluate_gsm8k_math(fn, force_cleanup=False, dataset='math')

def evaluate_champ(fn, ip='localhost', port=8001, generator_model_name=None):
    if 'OPENAI_API_KEY' not in os.environ:
        raise Exception('OPENAI_API_KEY is not set in the environment variables.')
    from copy import deepcopy as copy
    import champ_dataset
    from utils.vllm import VllmEndpointCaller
    SUMMARIZATION_PROMPT = 'Now, summarize the answer above in one sentence, without any intermediate steps or explanations.'
    GRADING_SYS_PROMPT = ("You are a math teacher and need to grade student's homework. "
                        'For each question, you need to determine the correctness of the '
                        'student answer, given the reference answer. You only need to '
                        'judge the correctness of the final answer, and should not consider '
                        'any reasoning or explanations given in the answer. Note that if '
                        'the student gives an obviously equivalent solution to the reference '
                        'answer (e.g., 1.5 vs 3/2 or a^2-b^2 vs (a+b)(a-b)), the answer should '
                        'be judged as correct. Your decision should be one of "Correct", '
                        '"Incorrect" or "Partially correct". There is no need to explain '
                        'your decision.')
    GRADING_USER_PROMPT = ('The question is:\n{problem_text}\n\n'
                        'The reference answer is:\n{ref_answer}\n\n'
                        'The student answer is:\n{model_answer}\n\n'
                        'Is the student answer correct, incorrect, or partially correct?')
    dataset_champ = champ_dataset.load('v0')

    def to_chat_input(sys_prompt, user_inputs, imputed_outputs):
        assert len(user_inputs) == len(imputed_outputs) + 1
        conversation = [dict(role='system', content=sys_prompt)]
        for u, i in zip(user_inputs, imputed_outputs):
            conversation.append(dict(role='user', content=u))
            conversation.append(dict(role='assistant', content=i))
        conversation.append(dict(role='user', content=user_inputs[-1]))
        return conversation

    response_data = []
    with open(fn, 'r') as f:
        for line in f:
            data = json.loads(line)
            response_data.append((data['query']['metadata']['problem_id'], data['responses'][0]['content']))

    # summarize responses
    generator_champ = champ_dataset.PromptGenerator(dataset_champ)
    champ_idx_to_problem_prompt = {}
    for problem_id, problem in dataset_champ.problems.items():
        sys_prompt, user_inputs, imputed_outputs = generator_champ.construct_prompt('0-Shot', problem) 
        messages = to_chat_input(sys_prompt, user_inputs, imputed_outputs)
        champ_idx_to_problem_prompt[problem_id] = messages
    conversations = []
    for problem_id, response in response_data:
        messages = champ_idx_to_problem_prompt[problem_id]
        conv = copy(messages)
        conv.append({'role': 'assistant', 'content': response})
        conv.append({'role': 'user', 'content': SUMMARIZATION_PROMPT})
        conversations.append(conv)
    caller = VllmEndpointCaller(ip=ip, port=port, model_name=generator_model_name, temperature=0.0, max_tokens=1024, top_p=1.0)
    summaries = caller.generate_batch(conversations)
    summaries = [(problem_id, summary) for (problem_id, _), summary in zip(response_data, summaries)]

    # grade summaries
    caller = VllmEndpointCaller(base_url='https://api.openai.com/v1', api_key=os.environ['OPENAI_API_KEY'], model_name='gpt-4o', temperature=0.0, max_tokens=1024, top_p=1.0)
    conversations = []
    for problem_id, summary in summaries:
        problem = dataset_champ.problems[problem_id]
        conversation = [
        dict(role='system', content=GRADING_SYS_PROMPT), 
        dict(role='user', content=GRADING_USER_PROMPT.format(
            problem_text=problem.text, 
            ref_answer=problem.answer, 
            model_answer=summary
            ))
        ]
        conversations.append(conversation)
    outputs = caller.generate_batch(conversations)

    return Counter([o.lower() for o in outputs])['correct'] / len(outputs)

def evaluate_humaneval_mbpp(fn, dataset=None):
    from evalplus.sanitize import script as sanitize
    from evalplus.evaluate import evaluate
    if dataset is None:
        assert 'humaneval' in fn or 'mbpp' in fn, 'dataset must be provided if fn does not contain mbpp or humaneval'
        if 'humaneval' in fn:
            dataset = 'humaneval'
        else:
            dataset = 'mbpp'
    raw_data = []
    for line in open(fn):
        data = json.loads(line)
        task_id = data['query']['metadata']['task_id']
        solution = data['responses'][0]['content']
        solution_id = task_id + '_sol'
        raw_data.append({'task_id': task_id, '_identifier': solution_id, 'solution': solution})
    sanitized_data = sanitize(raw_data)
    eval_result = evaluate(dataset, sanitized_data)
    all_results = [e[0]['plus_status'] for e in eval_result['eval'].values()]
    return Counter(all_results)['pass'] / len(all_results)

def evaluate_humaneval(fn):
    return evaluate_humaneval_mbpp(fn, 'humaneval')

def evaluate_mbpp(fn):
    return evaluate_humaneval_mbpp(fn, 'mbpp')

def evaluate_bigcodebench(fn):
    from bigcodebench.sanitize import script as sanitize
    from bigcodebench.evaluate import evaluate
    raw_data = []
    for line in open(fn):
        data = json.loads(line)
        task_id = data['query']['metadata']['task_id']
        solution = data['responses'][0]['content']
        solution_id = task_id + '_sol'
        raw_data.append({'task_id': task_id, '_identifier': solution_id, 'solution': solution})
    sanitized_data = sanitize(raw_data, calibrate=True)
    eval_result = evaluate('instruct', 'full', samples=sanitized_data, local_execute=True, pass_k="1", no_gt=True)
    all_results = [e[0]['status'] for e in eval_result['eval'].values()]
    return Counter(all_results)['pass'] / len(all_results)

def evaluate_alpacaeval(fn):
    if 'OPENAI_API_KEY' not in os.environ:
        raise Exception('OPENAI_API_KEY is not set in the environment variables.')
    import numpy as np
    from alpaca_eval import evaluate
    with open(fn, 'r') as f:
        data = [json.loads(line) for line in f.readlines()]
    formatted_data = [{'instruction': e['query']['content'], 'output': e['responses'][0]['content'], 'generator': 'default_generator', 
                        'dataset': 'default_dataset', 'datasplit': 'eval'} for e in data]
    eval_results = evaluate(model_outputs=formatted_data, is_return_instead_of_print=True, is_overwrite_leaderboard=True)[1]
    scores = [e['preference'] - 1 for e in eval_results if e['preference'] is not None]
    return float(np.mean(scores))

def evaluate_ifeval(fn):
    from instruction_following_eval.evaluation_main import evaluate
    with open(fn, 'r') as f:
        data = [json.loads(line) for line in f.readlines()]
    formatted_data = [{'query': e['query']['content'], 'response': e['responses'][0]['content']} for e in data]
    return evaluate(formatted_data)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--refinement-output-file', type=str, required=True)
    parser.add_argument('--dataset', type=str, required=False, default=None, choices=['gsm8k', 'math', 'champ', 'humaneval', 'mbpp', 'bigcodebench', 'alpacaeval', 'ifeval'])
    args = parser.parse_args()
    if args.dataset == 'None':
        for dataset in ['gsm8k', 'math', 'champ', 'humaneval', 'mbpp', 'bigcodebench', 'alpacaeval', 'ifeval']:
            if dataset in args.refinement_output_file:
                args.dataset = dataset
                break
        if args.dataset == None:
            raise ValueError('dataset name cannot be resolved from --refinement-output-fn.')
    func = eval(f'evaluate_{args.dataset}')
    score = func(args.refinement_output_file)
    print(f'Average score: {score * 100:0.2f}')
    os._exit(0)
