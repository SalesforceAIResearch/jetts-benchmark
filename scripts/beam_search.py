import json
import os, argparse
from vllm import SamplingParams
from jetts.judge.vllm_judge import VllmEndpointJudge
from jetts.judge_task.beam_search import BeamSearchTask
from jetts.prompts import load_prompter

def last_part(fn):
    return fn.strip('/').split('/')[-1]

def main(args):
    assert os.path.exists(args.input_dir), f'Input directory {args.input_dir} does not exist'
    if args.prompter is not None:
            prompter = load_prompter(args.prompter)
    else:
        prompter = None
    
    sampling_params = SamplingParams(n=1, temperature=0.0, top_p=1.0, max_tokens=1024)
    judge = VllmEndpointJudge(sampling_params=sampling_params, prompter=prompter, base_url=args.judge_base_url, 
                              api_key=args.api_key, model_name=args.judge_model, batch_size=args.vllm_batch_size)
    judge_model = judge.caller.model_name

    if args.final_rerank_method is None:
        args.final_rerank_method = args.rerank_method

    output_folder = os.path.join(args.output_dir, f'{last_part(judge_model)}_{last_part(args.input_dir)}_{args.rerank_method}_{args.final_rerank_method}')
    os.makedirs(output_folder, exist_ok=True)
    num_instances_map = {'gsm8k': 1319, 'math': 1324, 'champ': 270, 'bigcodebench': 1140, 'humaneval': 164, 'mbpp': 378}
    if args.dataset is None:
        for dataset in num_instances_map:
            if dataset in args.input_dir:
                args.dataset = dataset
                break
        assert args.dataset is not None, f'Dataset could not be inferred from input directory {args.input_dir}'
    num_instances = num_instances_map[args.dataset]
    if args.debug:
        only_first_n = 5
    else:
        only_first_n = None
    task = BeamSearchTask(input_dir=args.input_dir, output_dir=output_folder, 
                          num_instances=num_instances, lookahead=args.use_lookahead, 
                          rerank_params={'method': args.rerank_method}, final_rerank_params={'method': args.final_rerank_method}, only_first_n=only_first_n)
    task.run(judge)

    evaluate(output_folder, task.all_idxs)

def evaluate(output_folder, idxs):
    tot_count = 0
    tot_score = 0
    for idx in idxs:
        line = open(os.path.join(output_folder, f'{idx}.jsonl')).readlines()[0]
        data = json.loads(line)
        tot_count += 1
        tot_score += data['response_metadata']['score']
    print(f'Average score: {tot_score / tot_count * 100:0.2f}')

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', type=str, required=True, help='Path to input directory')
    parser.add_argument('--dataset', type=str, help='The dataset to use for the beam search. If not provided, will be inferred from the input directory')
    parser.add_argument('--judge-base-url', type=str, default='http://localhost:8000/v1', help='The base URL of the vllm reward model')
    parser.add_argument('--judge-model', type=str, help='The name of the judge model to use. Not needed if there is only one model running on the server')
    parser.add_argument('--prompter', type=str, help='The prompter to use for the judge. Will resolve automatically if not provided')
    parser.add_argument('--rerank-method', type=str, default='pairwise-rr', choices=['pairwise-rr', 'single-likert', 'single-additive'], 
                        help='The rerank method used for the step-wise reranking')
    parser.add_argument('--final-rerank-method', type=str, default=None, choices=['pairwise-rr', 'pairwise-ko', 'single-likert', 'single-additive', None], 
                        help='The rerank method used for the final reranking. Default to the same as rerank-method if not provided')
    parser.add_argument('--use-lookahead', action='store_true', help='Use lookahead for the beam search')
    parser.add_argument('--output-dir', type=str, default='outputs/beam_search', help='Path to the output directory')
    parser.add_argument('--api-key', type=str, default='sample-api-key', help='API key for the vllm judge endpoint')
    parser.add_argument('--vllm-batch-size', type=int, default=50, help='The batch size used for vllm inference')
    parser.add_argument('--debug', action='store_true', help='Debug mode, only the first 5 examples are processed')
    args = parser.parse_args()
    main(args)
