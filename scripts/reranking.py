import json
import os, argparse
from vllm import SamplingParams

from jetts.judge.vllm_judge import VllmEndpointJudge
from jetts.data import ModelResponseData
from jetts.judge_task.rerank import RerankTask
from jetts.prompts import load_prompter

def last_part(fn):
    return fn.strip('/').split('/')[-1]

def main(args):
    assert os.path.exists(args.data_file), f'Input file {args.data_file} does not exist'
    if args.prompter is not None:
            prompter = load_prompter(args.prompter)
    else:
        prompter = None
    
    sampling_params = SamplingParams(n=1, temperature=0.0, top_p=1.0, max_tokens=1024)
    judge = VllmEndpointJudge(sampling_params=sampling_params, prompter=prompter, base_url=args.judge_base_url, 
                              api_key=args.api_key, model_name=args.judge_model, batch_size=args.vllm_batch_size)
    judge_model = judge.caller.model_name

    os.makedirs(args.output_dir, exist_ok=True)
    out_fn = os.path.join(args.output_dir, f'{last_part(judge_model)}_{last_part(args.data_file).replace(".jsonl", "")}_{args.method}.jsonl')
    judging_data = ModelResponseData.from_jsonl(args.data_file)

    if args.debug:
        out_fn = out_fn.replace('.jsonl', '.debug.jsonl')
        judging_data = judging_data[:5]

    if os.path.isfile(out_fn) and not args.force_rerun:
        print(f'Output file {out_fn} exists! Skipping to evaluation')
        return evaluate(out_fn)
    
    task = RerankTask(judging_data, rerank_params={'method': args.method})
    task.run(judge, output_fn=out_fn, use_tqdm='both')

    return evaluate(out_fn)

def evaluate(out_fn):
    tot_count = 0
    tot_score = 0
    with open(out_fn, 'r') as f:
        for line in f:
            data = json.loads(line)
            tot_score += data['responses'][0]['metadata']['score']
            tot_count += 1
    print(f'Average score: {tot_score / tot_count * 100:0.2f}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-file', type=str, required=True, help='Path to the output generation file')
    parser.add_argument('--judge-base-url', type=str, default='http://localhost:8000/v1', help='The base URL of the vllm reward model')
    parser.add_argument('--judge-model', type=str, help='The name of the judge model to use. Not needed if there is only one model running on the server')
    parser.add_argument('--prompter', type=str, help='The prompter to use for the judge. Will resolve automatically if not provided')
    parser.add_argument('--method', type=str, default='pairwise-rr', choices=['pairwise-rr', 'pairwise-ko', 'single-likert', 'single-additive'], help='Reranking method')
    parser.add_argument('--output-dir', type=str, default='outputs/reranking', help='Path to the output directory')
    parser.add_argument('--api-key', type=str, default='sample-api-key', help='API key for the vllm judge endpoint')
    parser.add_argument('--vllm-batch-size', type=int, default=50, help='The batch size used for vllm inference')
    parser.add_argument('--debug', action='store_true', help='Debug mode, only the first 5 examples are processed')
    parser.add_argument('--force-rerun', action='store_true', help='Force rerun even if the output file already exists')
    args = parser.parse_args()
    main(args)
