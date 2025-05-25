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
"""
Preprocess the GSM8k dataset to parquet format
"""

import os
import datasets

from verl.utils.hdfs_io import copy, makedirs
import argparse

from verl.utils.reward_score.math import remove_boxed, last_boxed_only_string

from string import Template
from datasets import Dataset, load_dataset
import json

prompt_template = """${question} Let's think step by step and output the final answer within \\boxed{ }"""

def extract_solution(solution_str):
    return remove_boxed(last_boxed_only_string(solution_str))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_path",
        default="dataset/deepscaler_dataset"
    )
    parser.add_argument(
        "--local_dir",
        default="data/deepsacler_selective_token_deduplicated_r_prompt_shuffled",
    )
    parser.add_argument(
        "--hdfs_dir",
        default="hdfs/deepsacler_selective_token_deduplicated_r_prompt_shuffled",
    )
    args = parser.parse_args()

    train_question_list = []
    train_answer_list = []
    test_question_list = []
    test_answer_list = []
    train_dataloader_all = load_dataset(path=args.input_path,
                                    split="train",
                                    trust_remote_code=True)
    train_dataloader_all = train_dataloader_all.shuffle(seed=42)

    train_total_count = int(len(train_dataloader_all)*0.98)
    
    for i,data in enumerate(train_dataloader_all):
        if i <train_total_count:
            train_question_list.append(data["problem"])
            train_answer_list.append(data["answer"])
        elif i>train_total_count:
            test_question_list.append(data["problem"])
            test_answer_list.append(data["answer"])
    
    train_data_list = []
    for idx, (train_question, train_answer) in enumerate(
            zip(train_question_list, train_answer_list)):
        train_data_list.append({
            "data_source":
            "deepsacler_selective_token",
            "prompt": [{
                "role":
                "user",
                "content":
                Template(prompt_template).safe_substitute({"question":train_question}),
            }],
            "ability":
            "math",
            "reward_model": {
                "style": "rule",
                "ground_truth": train_answer
            },
            "extra_info": {
                "split": "train",
                "index": idx
            },
        })
    train_dataset = Dataset.from_list(train_data_list)
    
    test_data_list = []
    for idx, (test_question, test_answer) in enumerate(
            zip(test_question_list, test_answer_list)):
        test_data_list.append({
            "data_source":
            "deepsacler_validation",
            "prompt": [{
                "role":
                "user",
                "content":
                Template(prompt_template).safe_substitute({"question":test_question}),
            }],
            "ability":
            "math",
            "reward_model": {
                "style": "rule",
                "ground_truth": test_answer
            },
            "extra_info": {
                "split": "test",
                "index": idx
            },
        })
    test_dataset = Dataset.from_list(test_data_list)

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir
    train_dataset.to_parquet(os.path.join(local_dir, "train.parquet"))
    test_dataset.to_parquet(os.path.join(local_dir, "test.parquet"))

    if hdfs_dir is not None:
        copy(src=local_dir, dst=hdfs_dir)
