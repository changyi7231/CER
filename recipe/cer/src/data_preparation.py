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

import os
import argparse
import pandas
import datasets

from typing import List, Dict


qwen_prompt = "Please reason step by step, and put your final answer within \\boxed{}."
qwen_multi_choice_prompt = "Please reason step by step, and put your final answer within \\boxed{}. Please only provide the letter of the answer in the box."


def get_prompt(system_prompt: str, question: str) -> List[Dict[str, str]]:
    prompt = [{
        "role": "system",
        "content": "You are a helpful assistant."
        }, {
        "role": "user",
        "content": question + "\n" + system_prompt
        }]
    return prompt


def repeat_rows_in_parquet(input_dir: str, output_dir: str, n: int) -> None:
    df = pandas.read_parquet(input_dir)
    repeated_index = df.index.repeat(n)
    df_repeated = df.loc[repeated_index].reset_index(drop=True)
    
    # remove id column
    if "id" in df_repeated.columns:
        df_repeated = df_repeated.drop(columns=["id"])
    df_repeated.to_parquet(output_dir, index=False)
    print(f"Repeat dataset {n} times and save to: {output_dir}")


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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='')
    parser.add_argument('--save_dir', type=str, default='/save/data')
    parser.add_argument('--n_repeat', type=int, default=1)
    parser.add_argument('--data_source', type=str, default='TIGER-Lab/WebInstruct-verified',
                        choices=['TIGER-Lab/WebInstruct-verified', 'DigitalLearningGmbH/MATH-lighteval', 'math-ai/math500', 'math-ai/amc23', 'math-ai/aime24', 'math-ai/aime25', 'TIGER-Lab/MMLU-Pro', 'm-a-p/SuperGPQA'])
    args = parser.parse_args()
    
    if args.data_source == "TIGER-Lab/WebInstruct-verified":
        print(f"Loading the {args.data_source} dataset", flush=True)
        dataset = datasets.load_dataset(os.path.join(args.data_dir, args.data_source), trust_remote_code=True)
        train_dataset = dataset["train"]
        train_dataset = train_dataset.filter(
            lambda example: (
                example["category"] != "Mathematics"
                and example["difficulty"] == "University"
                )
            )
        print(f"Filtered train size: {len(train_dataset)}", flush=True)

        def make_map_fn(split):
            def process_fn(example, idx):
                question = example.pop("question")
                answer = example.pop("answer")
                ability = example.pop("category")
                prompt = get_prompt(qwen_prompt, question)
                data = {
                    "data_source": args.data_source,
                    "prompt": prompt,
                    "ability": ability,
                    "reward_model": {
                        "style": "rule",
                        "ground_truth": answer
                    },
                    "extra_info": {
                        "split": split,
                        "index": idx
                    }
                }
                return data
            return process_fn

        train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True)
        save_dir = os.path.join(args.save_dir, args.data_source)
        train_dataset.to_parquet(os.path.join(save_dir, "train.parquet"))
        print(train_dataset[0])
        
        repeat_rows_in_parquet(os.path.join(save_dir, "train.parquet"), os.path.join(save_dir, "train_repeated.parquet"), args.n_repeat)
        
    elif args.data_source == "DigitalLearningGmbH/MATH-lighteval":
        print(f"Loading the {args.data_source} dataset", flush=True)
        dataset = datasets.load_dataset(os.path.join(args.data_dir, args.data_source), trust_remote_code=True)
        train_dataset = dataset["train"]
        test_dataset = dataset["test"]
        def make_map_fn(split):
            def process_fn(example, idx):
                question = example.pop("problem")
                solution = example.pop("solution")
                answer = extract_boxed_answer(solution)
                prompt = get_prompt(qwen_prompt, question)
                data = {
                    "data_source": args.data_source,
                    "prompt": prompt,
                    "ability": "math",
                    "reward_model": {
                        "style": "rule",
                        "ground_truth": answer
                    },
                    "extra_info": {
                        "split": split,
                        "index": idx
                    }
                }
                return data
            return process_fn

        train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True)
        test_dataset = test_dataset.map(function=make_map_fn("test"), with_indices=True)
        save_dir = os.path.join(args.save_dir, args.data_source)
        train_dataset.to_parquet(os.path.join(save_dir, "train.parquet"))
        test_dataset.to_parquet(os.path.join(save_dir, "test.parquet"))
        print(train_dataset[0])
        print(test_dataset[0])
        
        repeat_rows_in_parquet(os.path.join(save_dir, "train.parquet"), os.path.join(save_dir, "train_repeated.parquet"), args.n_repeat)
        repeat_rows_in_parquet(os.path.join(save_dir, "test.parquet"), os.path.join(save_dir, "test_repeated.parquet"), args.n_repeat)
    
    elif args.data_source == "math-ai/math500":
        print(f"Loading the {args.data_source} dataset", flush=True)
        dataset = datasets.load_dataset(os.path.join(args.data_dir, args.data_source), trust_remote_code=True)
        test_dataset = dataset["test"]
        
        def make_map_fn(split):
            def process_fn(example, idx):
                question = example.pop("problem")
                answer = example.pop("answer")
                prompt = get_prompt(qwen_prompt, question)
                data = {
                    "data_source": args.data_source,
                    "prompt": prompt,
                    "ability": "math",
                    "reward_model": {
                        "style": "rule",
                        "ground_truth": answer
                    },
                    "extra_info": {
                        "split": split,
                        "index": idx
                    }
                }
                return data
            return process_fn

        test_dataset = test_dataset.map(function=make_map_fn("test"), with_indices=True)
        save_dir = os.path.join(args.save_dir, args.data_source)
        test_dataset.to_parquet(os.path.join(save_dir, "test.parquet"))
        print(test_dataset[0])
        
        repeat_rows_in_parquet(os.path.join(save_dir, "test.parquet"), os.path.join(save_dir, "test_repeated.parquet"), args.n_repeat)
               
    elif args.data_source == "math-ai/amc23":
        print(f"Loading the {args.data_source} dataset", flush=True)
        dataset = datasets.load_dataset(os.path.join(args.data_dir, args.data_source), trust_remote_code=True)
        test_dataset = dataset["test"]

        def make_map_fn(split):
            def process_fn(example, idx):
                question = example.pop("question")
                answer = example.pop("answer")
                prompt = get_prompt(qwen_prompt, question)
                data = {
                    "data_source": args.data_source,
                    "prompt": prompt,
                    "ability": "math",
                    "reward_model": {
                        "style": "rule",
                        "ground_truth": answer
                    },
                    "extra_info": {
                        "split": split,
                        "index": idx
                    }
                }
                return data
            return process_fn

        test_dataset = test_dataset.map(function=make_map_fn("test"), with_indices=True)
        save_dir = os.path.join(args.save_dir, args.data_source)
        test_dataset.to_parquet(os.path.join(save_dir, "test.parquet"))
        print(test_dataset[0])
        
        repeat_rows_in_parquet(os.path.join(save_dir, "test.parquet"), os.path.join(save_dir, "test_repeated.parquet"), args.n_repeat)
        
    elif args.data_source == "math-ai/aime24":
        print(f"Loading the {args.data_source} dataset", flush=True)
        dataset = datasets.load_dataset(os.path.join(args.data_dir, args.data_source), trust_remote_code=True)
        test_dataset = dataset["test"]

        def make_map_fn(split):
            def process_fn(example, idx):
                question = example.pop("problem")
                answer = example.pop("solution")
                prompt = get_prompt(qwen_prompt, question)
                data = {
                    "data_source": args.data_source,
                    "prompt": prompt,
                    "ability": "math",
                    "reward_model": {
                        "style": "rule",
                        "ground_truth": extract_boxed_answer(answer)
                    },
                    "extra_info": {
                        "split": split,
                        "index": idx
                    }
                }
                return data
            return process_fn

        test_dataset = test_dataset.map(function=make_map_fn("test"), with_indices=True)
        save_dir = os.path.join(args.save_dir, args.data_source)
        test_dataset.to_parquet(os.path.join(save_dir, "test.parquet"))
        print(test_dataset[0])
        
        repeat_rows_in_parquet(os.path.join(save_dir, "test.parquet"), os.path.join(save_dir, "test_repeated.parquet"), args.n_repeat)
        
    elif args.data_source == "math-ai/aime25":
        print(f"Loading the {args.data_source} dataset", flush=True)
        dataset = datasets.load_dataset(os.path.join(args.data_dir, args.data_source), trust_remote_code=True)
        test_dataset = dataset["test"]

        def make_map_fn(split):
            def process_fn(example, idx):
                question = example.pop("problem")
                answer = example.pop("answer")
                prompt = get_prompt(qwen_prompt, question)
                data = {
                    "data_source": args.data_source,
                    "prompt": prompt,
                    "ability": "math",
                    "reward_model": {
                        "style": "rule",
                        "ground_truth": answer
                    },
                    "extra_info": {
                        "split": split,
                        "index": idx
                    }
                }
                return data
            return process_fn

        test_dataset = test_dataset.map(function=make_map_fn("test"), with_indices=True)
        save_dir = os.path.join(args.save_dir, args.data_source)
        test_dataset.to_parquet(os.path.join(save_dir, "test.parquet"))
        print(test_dataset[0])
        
        repeat_rows_in_parquet(os.path.join(save_dir, "test.parquet"), os.path.join(save_dir, "test_repeated.parquet"), args.n_repeat)

    elif args.data_source == "TIGER-Lab/MMLU-Pro":
        print(f"Loading the {args.data_source} dataset", flush=True)
        dataset = datasets.load_dataset(os.path.join(args.data_dir, args.data_source), trust_remote_code=True)
        test_dataset = dataset["test"]
        
        def make_map_fn(split):
            def process_fn(example, idx):
                letters = "ABCDEFGHIJ"
                option_lines = []
                for i, opt_text in enumerate(example["options"]):
                    if i >= len(letters):
                        break
                    option_lines.append(f"{letters[i]}: {opt_text}")
                options_block = "\n".join(option_lines)
                question = example["question"]+ "\n" + options_block
                answer = example["answer"]
                prompt = get_prompt(qwen_multi_choice_prompt, question)
                data = {
                    "data_source": args.data_source,
                    "prompt": prompt,
                    "ability": example["category"],
                    "reward_model": {
                        "style": "rule",
                        "ground_truth": answer},
                    "extra_info": {
                        "split": split,
                        "index": idx
                        }
                    }
                return data
            return process_fn

        test_dataset = test_dataset.map(function=make_map_fn("test"), with_indices=True)
        save_dir = os.path.join(args.save_dir, args.data_source)
        test_dataset.to_parquet(os.path.join(save_dir, 'test.parquet'))
        print(test_dataset[0])
        
        repeat_rows_in_parquet(os.path.join(save_dir, 'test.parquet'), os.path.join(save_dir, 'test_repeated.parquet'), args.n_repeat)
       
    elif args.data_source == "m-a-p/SuperGPQA":
        print(f"Loading the {args.data_source} dataset", flush=True)
        dataset = datasets.load_dataset(os.path.join(args.data_dir, args.data_source), trust_remote_code=True)
        test_dataset = dataset["train"]
        
        def make_map_fn(split):
            def process_fn(example, idx):
                letters = "ABCDEFGHIJ"
                option_lines = []
                for i, opt_text in enumerate(example["options"]):
                    if i >= len(letters):
                        break
                    option_lines.append(f"{letters[i]}: {opt_text}")
                options_block = "\n".join(option_lines)
                question = example["question"]+ "\n" + options_block
                answer = example["answer_letter"]
                prompt = get_prompt(qwen_multi_choice_prompt, question)
                data = {
                    "data_source": args.data_source,
                    "prompt": prompt,
                    "ability": example["field"],
                    "reward_model": {
                        "style": "rule",
                        "ground_truth": answer},
                    "extra_info": {
                        "split": split,
                        "index": idx
                        }
                    }
                return data
            return process_fn

        test_dataset = test_dataset.map(function=make_map_fn("test"), with_indices=True)
        save_dir = os.path.join(args.save_dir, args.data_source)
        test_dataset.to_parquet(os.path.join(save_dir, 'test.parquet'))
        print(test_dataset[0])
        
        repeat_rows_in_parquet(os.path.join(save_dir, 'test.parquet'), os.path.join(save_dir, 'test_repeated.parquet'), args.n_repeat)

    else:
        raise NotImplementedError
