# coding=utf-8
# Copyright 2024 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Binary of evaluating instruction following. See README.md."""
import dataclasses
import json
import numpy as np
from typing import Dict, Optional, Union
from instruction_following_eval import instructions_registry

@dataclasses.dataclass
class InputExample:
  key: int
  instruction_id_list: list[str]
  prompt: str
  kwargs: list[Dict[str, Optional[Union[str, int]]]]

@dataclasses.dataclass
class OutputExample:
  instruction_id_list: list[str]
  prompt: str
  response: str
  follow_all_instructions: bool
  follow_instruction_list: list[bool]

def read_prompt_list(input_jsonl_filename, responses):
  """Read inputs from jsonl."""
  inputs = []
  prompt_to_inputs = {}
  with open(input_jsonl_filename, "r") as f:
    for l in f:
      example = json.loads(l)
      prompt_to_inputs[example['prompt']] = InputExample(key=example["key"],
                      instruction_id_list=example["instruction_id_list"],
                      prompt=example["prompt"],
                      kwargs=example["kwargs"])
    
  seen_queries = set()
  for example in responses:
    if example['query'] not in seen_queries:
      inputs.append(prompt_to_inputs[example["query"]])
      seen_queries.add(example['query'])

  return inputs

def test_instruction_following_strict(
    inp,
    prompt_to_response,
):
  """Tests response to see if instrutions are followed."""
  responses = prompt_to_response[inp.prompt]
  outputs = []
  for response in responses:
    instruction_list = inp.instruction_id_list
    is_following_list = []

    for index, instruction_id in enumerate(instruction_list):
      instruction_cls = instructions_registry.INSTRUCTION_DICT[instruction_id]
      instruction = instruction_cls(instruction_id)

      args_desc = inp.kwargs[index]
      args_desc = {k:v for k, v in args_desc.items() if v is not None}
      instruction.build_description(**args_desc)
      args = instruction.get_instruction_args()
      if args and "prompt" in args:
        instruction.build_description(prompt=inp.prompt)

      if response.strip() and instruction.check_following(response):
        is_following_list.append(True)
      else:
        is_following_list.append(False)

    outputs.append(OutputExample(
        instruction_id_list=inp.instruction_id_list,
        prompt=inp.prompt,
        response=response,
        follow_all_instructions=all(is_following_list),
        follow_instruction_list=is_following_list,
    ))

  return outputs


def test_instruction_following_loose(
    inp,
    prompt_to_response,
):
  """Tests response for an upper bound for following instructions."""
  responses = prompt_to_response[inp.prompt]
  outputs = []
  for response in responses:
    r = response.split("\n")
    response_remove_first = "\n".join(r[1:]).strip()
    response_remove_last = "\n".join(r[:-1]).strip()
    response_remove_both = "\n".join(r[1:-1]).strip()
    revised_response = response.replace("*", "")
    revised_response_remove_first = response_remove_first.replace("*", "")
    revised_response_remove_last = response_remove_last.replace("*", "")
    revised_response_remove_both = response_remove_both.replace("*", "")
    all_responses = [
        response,
        revised_response,
        response_remove_first,
        response_remove_last,
        response_remove_both,
        revised_response_remove_first,
        revised_response_remove_last,
        revised_response_remove_both,
    ]
    instruction_list = inp.instruction_id_list
    is_following_list = []

    for index, instruction_id in enumerate(instruction_list):
      instruction_cls = instructions_registry.INSTRUCTION_DICT[instruction_id]
      instruction = instruction_cls(instruction_id)

      args_desc = inp.kwargs[index]
      args_desc = {k:v for k, v in args_desc.items() if v is not None}
      instruction.build_description(**args_desc)
      args = instruction.get_instruction_args()
      if args and "prompt" in args:
        instruction.build_description(prompt=inp.prompt)

      is_following = False
      for r in all_responses:
        if r.strip() and instruction.check_following(r):
          is_following = True
          break

      is_following_list.append(is_following)

    outputs.append(OutputExample(
        instruction_id_list=inp.instruction_id_list,
        prompt=inp.prompt,
        response=response,
        follow_all_instructions=all(is_following_list),
        follow_instruction_list=is_following_list,
    ))
  return outputs


def read_prompt_to_response_dict(responses):
  """Creates dictionary matching prompt and response."""
  return_dict = {}
  for example in responses:
    if example["query"] in return_dict:
      return_dict[example["query"]].append(example["response"])
    else:
      return_dict[example["query"]] = example["response"]
      if isinstance(example["response"], str):
        return_dict[example["query"]] = [example["response"]] # list now
  return return_dict

def compute_counts(example):
  follow_instruction_list = example.follow_instruction_list
  instruction_id_list = example.instruction_id_list

  prompt_total = 1
  prompt_correct = 0
  if all(follow_instruction_list):
    prompt_correct += 1

  instruction_total = len(instruction_id_list)
  instruction_correct = sum(follow_instruction_list)

  counts = {
    'prompt_correct': prompt_correct,
    'prompt_total': prompt_total,
    'instruction_correct': instruction_correct,
    'instruction_total': instruction_total,
  }
  return counts

# custom implemented based on multiple outputs
def compute_metrics(outputs):
  # for each output, we want to compute all prompt based and instruction based scores
  outputs_counts = [[compute_counts(o) for o in ol] for ol in outputs]
  instruction_total = sum([o[0]['instruction_total'] for o in outputs_counts])
  prompt_total = sum([o[0]['prompt_total'] for o in outputs_counts])

  # average perf
  avg_prompt_correct = float(sum([np.average([o['prompt_correct'] for o in oc_lst]) for oc_lst in outputs_counts]))
  avg_instr_correct = float(sum([np.average([o['instruction_correct'] for o in oc_lst]) for oc_lst in outputs_counts]))
  
  metrics = {
    'avg': {
      'prompt_accuracy': float(avg_prompt_correct / prompt_total),
      'instruction_accuracy': float(avg_instr_correct / instruction_total),
    }
  }

  if not all(len(oc_lst) == 1 for oc_lst in outputs):
    min_prompt_correct = float(sum([np.min([o['prompt_correct'] for o in oc_lst]) for oc_lst in outputs_counts]))
    max_prompt_correct = float(sum([np.max([o['prompt_correct'] for o in oc_lst]) for oc_lst in outputs_counts]))

    min_instr_correct = float(sum([np.min([o['instruction_correct'] for o in oc_lst]) for oc_lst in outputs_counts]))
    max_instr_correct = float(sum([np.max([o['instruction_correct'] for o in oc_lst]) for oc_lst in outputs_counts]))

    metrics['min'] = {
      'prompt_accuracy': float(min_prompt_correct / prompt_total),
      'instruction_accuracy': float(min_instr_correct / instruction_total),
    }

    metrics['max'] = {
      'prompt_accuracy': float(max_prompt_correct / prompt_total),
      'instruction_accuracy': float(max_instr_correct / instruction_total),
    }

    num_responses_per_sample = [len(o) for o in outputs]
    metrics['num_responses_statistics'] = {
        'min': min(num_responses_per_sample),
        'max': max(num_responses_per_sample),
        'avg': float(np.average(num_responses_per_sample))
    }
  return metrics



def evaluate(responses):
  inputs = read_prompt_list('./instruction_following_eval/data/input_data.jsonl', responses)
  prompt_to_response = read_prompt_to_response_dict(responses)

  # get instruction following results
  save_metrics = {}
  for func, output_file_name in [
      (test_instruction_following_strict, "eval_results_strict"),
      (test_instruction_following_loose, "eval_results_loose"),
  ]:
    outputs = []
    for inp in inputs:
      outputs.append(func(inp, prompt_to_response))

    metrics_out = compute_metrics(outputs)
    save_key = 'loose'
    if 'eval_results_strict' in output_file_name:
      save_key = 'strict'
    save_metrics[save_key] = metrics_out

  return save_metrics['strict']['avg']['prompt_accuracy']
