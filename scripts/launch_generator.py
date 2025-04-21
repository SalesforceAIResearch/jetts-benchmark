import argparse
import subprocess

generator_name_mapping = {'llama8b': 'meta-llama/Llama-3.1-8B-Instruct', 
                      'llama70b': 'meta-llama/Llama-3.1-70B-Instruct',
                      'qwen32b': 'Qwen/Qwen2.5-32B-Instruct',
                      'qwen72b': 'Qwen/Qwen2.5-72B-Instruct'}

parser = argparse.ArgumentParser()
parser.add_argument('--generator-model', type=str, required=True)
parser.add_argument('--num-gpus', type=int, default=1)
parser.add_argument('--max-model-len', type=int, default=8192)
parser.add_argument('--gpu-memory-utilization', type=float, default=0.8)
parser.add_argument('--port', type=int, default=8001)

args = parser.parse_args()

cmd = ['vllm', 'serve', generator_name_mapping.get(args.generator_model.lower(), args.generator_model), f'--tensor-parallel-size={args.num_gpus}', f'--max-model-len={args.max_model_len}',
        f'--gpu-memory-utilization={args.gpu_memory_utilization}', f'--port={args.port}', '--disable-log-stats', '--disable-log-requests']
subprocess.run(cmd)
