# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

import torch

from typing import Any, Dict, Union
from collections import defaultdict
from verl import DataProto
from latex2sympy2_extended import NormalizationConfig
from math_verify import LatexExtractionConfig, parse, verify


def math_accuracy_reward(solution: str, golden_answer: str) -> Dict[str, float | str]:
    """Reward function that checks whether the answer is equivalent to the golden answer."""
    extracted_golden_answer = parse(
        "\\boxed{" + golden_answer + "}",
        extraction_mode="first_match",
        fallback_mode="first_match",
        extraction_config=[
            LatexExtractionConfig(
                normalization_config=NormalizationConfig(
                    basic_latex=True,
                    units=True,
                    malformed_operators=True,
                    nits=True,
                    boxed="all",
                ),
                # Ensures that boxed is tried first
                boxed_match_priority=0,
                try_extract_without_anchor=True,
            )
        ],
    )
    if len(extracted_golden_answer) == 0:
        print(f"fail to extract golden answer {golden_answer}")
        reward = 0.0
        result = {
            "score": reward,
            "acc": reward,
            "pred": solution[-512:]
        }
        return result
    
    extracted_answer = parse(
        solution[-512:],
        extraction_mode="first_match",
        fallback_mode="first_match",
        extraction_config=[
            LatexExtractionConfig(
                normalization_config=NormalizationConfig(
                    basic_latex=True,
                    units=True,
                    malformed_operators=True,
                    nits=True,
                    boxed="all",
                ),
                # Ensures that boxed is tried first
                boxed_match_priority=0,
                try_extract_without_anchor=True,
            )
        ],
    )
    if len(extracted_answer) == 0:
        # print(f"fail to extract answer: {solution}")
        reward = 0.0
        result = {
            "score": reward,
            "acc": reward,
            "pred": solution[-512:],
        }
        return result
    # Reward 1 if the answer is equivalent to the golden answer, 0 otherwise.
    reward = float(verify(extracted_golden_answer[0], extracted_answer[0]))
    result = {
        "score": reward,
        "acc": reward,
        "pred": str(extracted_answer[-1]),
    }
    return result


def extract_boxed_answer(text: str) -> str:
    key = r"\boxed{"
    n = len(text)
    answer = ""
    for start in range(n):
        if not text.startswith(key, start):
            continue
        brace_level = 1
        content_start = start + len(key)
        for i in range(content_start, n):
            if text[i] == "{":
                brace_level += 1
            elif text[i] == "}":
                brace_level -= 1
            if brace_level == 0:
                answer = text[content_start:i]
                break
    return answer


def exact_match(solution: str, ground_truth: str) -> Dict[str, float | str]:
    extracted_answer = extract_boxed_answer(solution)
    score = 1.0 if extracted_answer == ground_truth else 0.0
    return {
        "score": score,
        "acc": score,
        "pred": extracted_answer
    }


def compute_score(data_source: str, solution: str, ground_truth: str) -> Dict[str, float | str]:
    if data_source in ['TIGER-Lab/WebInstruct-verified', 'DigitalLearningGmbH/MATH-lighteval', "math-ai/math500", "math-ai/amc23", "math-ai/aime24", "math-ai/aime25"]:
        result = math_accuracy_reward(solution, ground_truth)
    elif data_source in ["m-a-p/SuperGPQA", "TIGER-Lab/MMLU-Pro"]:
        result = exact_match(solution, ground_truth)
    else:
        raise ValueError("wrong data source")
    return result


class RLRewardManager:
    def __init__(self, tokenizer, reward_fn_key="data_source"):
        self.tokenizer = tokenizer
        self.reward_fn_key = reward_fn_key

    def __call__(self, data: DataProto, return_dict: bool = True, num_examine: int = 1, rewards: torch.Tensor = None) -> Union[torch.Tensor, Dict[str, Any]]:
        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)
        already_print_data_sources = {}
        
        for i in range(len(data)):
            data_item = data[i]
            prompt_ids = data_item.batch["prompts"]
            prompt_length = prompt_ids.shape[-1]
            valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum().item()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum().item()
            valid_response_ids = response_ids[:valid_response_length]

            question_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            solution_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
            
            ground_truth_str = data_item.non_tensor_batch["reward_model"]["ground_truth"]
            data_source = data_item.non_tensor_batch[self.reward_fn_key]
            
            if rewards is not None:
                score = rewards[i].item()
                reward = score
            else:
                score = compute_score(
                    data_source=data_source,
                    solution=solution_str,
                    ground_truth=ground_truth_str
                    )

                if isinstance(score, dict):
                    reward = score["score"]
                    for key, value in score.items():
                        reward_extra_info[key].append(value)
                else:
                    reward = score
            reward_tensor[i, valid_response_length - 1] = reward
            
            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0
            if already_print_data_sources[data_source] < num_examine:
                already_print_data_sources[data_source] += 1
                print("[data source]", data_source)
                print("[question]", question_str)
                print("[solution]", solution_str)
                print("[ground truth]", ground_truth_str)
                if isinstance(score, dict):
                    for key, value in score.items():
                        print(f"[{key}]", value)
                else:
                    print("[score]", score)
        
        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor
