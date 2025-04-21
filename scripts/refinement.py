import os, argparse
from vllm import SamplingParams
from jetts.judge.vllm_judge import VllmEndpointJudge
from jetts.data import ModelResponseData
from jetts.prompts import load_prompter
from jetts.refiner import VllmEndpointRefiner
from jetts.judge_task import SingleInstanceRefineTask

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
    refiner = VllmEndpointRefiner(sampling_params=sampling_params, base_url=args.refiner_base_url, 
                            api_key=args.api_key, model_name=args.refiner_model, batch_size=args.vllm_batch_size)
    refiner_model = refiner.caller.model_name

    os.makedirs(args.output_dir, exist_ok=True)
    out_fn = os.path.join(args.output_dir, f'{last_part(judge_model)}_{last_part(refiner_model)}_{last_part(args.data_file).replace(".jsonl", "")}_{args.rerank_method}.jsonl')
    judging_data = ModelResponseData.from_jsonl(args.data_file, max_keep=args.num_seed_responses)

    if args.debug:
        out_fn = out_fn.replace('.jsonl', '.debug.jsonl')
        judging_data.data = judging_data.data[:5]

    if os.path.isfile(out_fn) and not args.force_rerun:
        print(f'Output file {out_fn} exists! Skipping to evaluation')
        return evaluate(out_fn)

    task = SingleInstanceRefineTask(judging_data, max_iterations=args.num_iterations, rerank_params={'method': args.rerank_method})
    task.run(judge, refiner, return_all_refinements=True, refine_tqdm='both', rerank_tqdm='dataset', reranking_output_fn=out_fn)

    evaluate(out_fn)

def evaluate(out_fn):
    pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-file', type=str, required=True, help='Path to the output generation file')
    parser.add_argument('--num-iterations', type=int, default=9, help='Number of iterations for refinement')
    parser.add_argument('--num-seed-responses', type=int, default=1, help='Number of seed responses to use for refinement')
    parser.add_argument('--rerank-method', type=str, default='pairwise-rr', 
                        choices=['pairwise-rr', 'pairwise-ko', 'single-likert', 'single-additive'], help='The rank_all method used by the final reranking step')
    parser.add_argument('--judge-base-url', type=str, default='http://localhost:8000/v1', help='The base URL of the vllm reward model')
    parser.add_argument('--judge-model', type=str, help='The name of the judge model to use. Not needed if there is only one model running on the server')
    parser.add_argument('--refiner-base-url', type=str, default='http://localhost:8001/v1', help='The base URL of the vllm refiner model')
    parser.add_argument('--refiner-model', type=str, help='The name of the refiner model to use. Not needed if there is only one model running on the server')
    parser.add_argument('--prompter', type=str, help='The prompter to use for the judge. Will resolve automatically if not provided')
    parser.add_argument('--output-dir', type=str, default='outputs/refinement', help='Path to the output directory')
    parser.add_argument('--api-key', type=str, default='sample-api-key', help='API key for the vllm judge endpoint')
    parser.add_argument('--vllm-batch-size', type=int, default=50, help='The batch size used for vllm inference')
    parser.add_argument('--debug', action='store_true', help='Debug mode, only the first 5 examples are processed')
    parser.add_argument('--force-rerun', action='store_true', help='Force rerun even if the output file already exists')
    args = parser.parse_args()
    main(args)
