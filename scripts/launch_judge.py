import argparse
import subprocess

judge_name_mapping = {'prom7b': 'prometheus-eval/prometheus-7b-v2.0', 
                      'sc8b': 'Skywork/Skywork-Critic-Llama-3.1-8B',
                      'ob8b': 'NCSOFT/Llama-3-OffsetBias-8B',
                      'thm8b': 'PKU-ONELab/Themis',
                      'prom8x7b': 'prometheus-eval/prometheus-8x7b-v2.0', 
                      'sc70b': 'Skywork/Skywork-Critic-Llama-3.1-70B',
                      'ste70b': 'facebook/Self-taught-evaluator-llama3.1-70B',
                      'llama8b': 'meta-llama/Llama-3.1-8B-Instruct'}

parser = argparse.ArgumentParser()
parser.add_argument('--judge-model', type=str, required=True)
parser.add_argument('--num-gpus', type=int, default=1)
parser.add_argument('--max-model-len', type=int, default=8192)
parser.add_argument('--gpu-memory-utilization', type=float, default=0.8)
parser.add_argument('--port', type=int, default=8000)

args = parser.parse_args()

cmd = ['vllm', 'serve', judge_name_mapping.get(args.judge_model.lower(), args.judge_model), f'--tensor-parallel-size={args.num_gpus}', f'--max-model-len={args.max_model_len}',
        f'--gpu-memory-utilization={args.gpu_memory_utilization}', f'--port={args.port}', '--disable-log-stats', '--disable-log-requests']
subprocess.run(cmd)
