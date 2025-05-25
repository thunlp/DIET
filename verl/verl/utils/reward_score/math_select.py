# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
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
# Adapted from https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/hendrycks_math/utils.py

import torch
import math
import re
import random
import numpy as np
def detect_repetition_with_hash(text, window_size=10):
    """
    Use hashing to efficiently detect repetitions
    """
    words = text.split()
    if len(words) <= window_size:
        return 0.0
    
    seen_hashes = set()
    repetitions = 0
    
    for i in range(len(words) - window_size + 1):
        # Get window and its hash
        window = tuple(words[i:i+window_size])
        window_hash = hash(window)
        
        # Check if we've seen this hash before
        if window_hash in seen_hashes:
            # repetitions += 1
            if repetitions >= 6:
                break
        else:
            seen_hashes.add(window_hash)
    last_boxed_position = 0
    box_count = 0
    try:
        last_boxed_position =list(re.finditer(r"\\boxed", text))[-1].start()
        box_count = len(re.findall(r"\\boxed",text))
    except:
        pass
    repetition_score = -1 if repetitions >= 6 or box_count>2 or (last_boxed_position/len(text)) <0.6 else 0
    return repetition_score

def compute_score(solution_str, ground_truth, rollout_results,
                  tokenizer,use_token:bool=False,use_adaptive_token:bool=False,use_anneal_token:bool=False,training_step:int =0,num_steps_per_half_period:int =0,use_target_token:bool=False,token_score_weight:float =0.025) -> float:
    if detect_repetition_with_hash(solution_str,window_size=10) ==-1:
        return -1,0,-1,0
    
    token_score_weight = token_score_weight
    token_score = 0
    valid_rollout_results = []
    answer_list =[]
    if use_token:
        # token_score_weight = token_score_weight
        if rollout_results[0].non_tensor_batch["extra_info"]["split"] != "test":
            for i in range(len(rollout_results)):
                # result = rollout_results[i]
                prompt_ids = rollout_results[i].batch["prompts"]

                prompt_length = prompt_ids.shape[-1]

                valid_prompt_length = rollout_results[i].batch["attention_mask"][:prompt_length].sum()
                valid_prompt_ids = prompt_ids[-valid_prompt_length:]

                response_ids = rollout_results[i].batch["responses"]
                valid_response_length = rollout_results[i].batch["attention_mask"][
                    prompt_length:].sum()
                valid_response_ids = response_ids[:valid_response_length]

                # decode
                # sequences = torch.cat((valid_prompt_ids, valid_response_ids))
                sequences_str = tokenizer.decode(valid_response_ids)
                try:
                    answer_list.append(remove_boxed(last_boxed_only_string(sequences_str)))
                except:
                    answer_list.append(None)
                try :
                    # 只在回答正确的sample中计算token_score
                    if is_equiv(answer_list[-1],ground_truth):
                        valid_rollout_results.append(sequences_str)
                except Exception as e:
                    print(e)
            if len(valid_rollout_results) >0:
                valid_token_list = [
                    len(tokenizer.tokenize(result))
                    for result in valid_rollout_results
                ]
                min_len = min(valid_token_list)
                max_len = max(valid_token_list)
                if min_len != max_len:
                    token_score = 0.5 - (
                        (len(tokenizer.tokenize(solution_str)) - min_len) /
                        (max_len - min_len))
                else:
                    token_score = 0
            else:
                token_score = 0
    if use_adaptive_token:
        # token_score_weight = token_score_weight
        right_count = 0
        for answer in answer_list:
            try:
                if is_equiv(answer,ground_truth):
                    right_count+=1
            except Exception as e:
                print(e)
        token_score_weight = token_score_weight*(right_count/len(answer_list))
      
    if use_anneal_token:
        omiga = (2*math.pi) / (num_steps_per_half_period*2)
        cosine_weight = (math.cos(omiga*training_step) + 1) / 2
        token_score_weight*=cosine_weight
    
    if use_target_token:
        right_count = 0 
        # token_score_weight = token_score_weight
        for answer in answer_list:
            try:
                if is_equiv(answer,ground_truth):
                    right_count+=1
            except Exception as e:
                print(e)
        error_count = len(rollout_results) - right_count
        error_ratio = error_count / len(rollout_results)
        target_token_num = random.randint(int(8192*(error_ratio-0.1)),
                                          int(8192*error_ratio))
        if error_count == 0:
            target_token_num = 0
        
        res_target_list = []

        if rollout_results[0].non_tensor_batch["extra_info"]["split"] != "test":
            for i in range(len(rollout_results)):
                # result = rollout_results[i]
                prompt_ids = rollout_results[i].batch["prompts"]

                prompt_length = prompt_ids.shape[-1]

                valid_prompt_length = rollout_results[i].batch["attention_mask"][:prompt_length].sum()
                valid_prompt_ids = prompt_ids[-valid_prompt_length:]

                response_ids = rollout_results[i].batch["responses"]
                valid_response_length = rollout_results[i].batch["attention_mask"][
                    prompt_length:].sum()
                valid_response_ids = response_ids[:valid_response_length]

                # decode
                # sequences = torch.cat((valid_prompt_ids, valid_response_ids))
                sequences_str = tokenizer.decode(valid_response_ids)
                try:
                    answer_list.append(remove_boxed(last_boxed_only_string(sequences_str)))
                except:
                    answer_list.append(None)
                res_target_list.append(
                    max(len(tokenizer.tokenize(sequences_str))-target_token_num, 0)
                )
                
        target_mean = np.mean(res_target_list)
        target_std = np.std(res_target_list)
        if target_std != 0:
            token_score = - ((max(len(tokenizer.tokenize(solution_str))-target_token_num, 0) - target_mean) / target_std)
        else:
            token_score = 0
    
    retval = 0.0
    acc = 0.0
    try:
        string_in_last_boxed = last_boxed_only_string(solution_str)
        if string_in_last_boxed is not None:
            answer = remove_boxed(string_in_last_boxed)
            if is_equiv(answer, ground_truth):
                acc = 1
                retval = 1.0 + token_score_weight*token_score
            else:
                retval = 0
                token_score = 0
    except Exception as e:
        print(e)
        token_score = 0
    return retval, acc,token_score,token_score_weight


# string normalization from https://github.com/EleutherAI/lm-evaluation-harness/blob/master/lm_eval/tasks/hendrycks_math.py
def is_equiv(str1, str2, verbose=False):
    
    if str1 is None and str2 is None:
        print("WARNING: Both None")
        return True
    if str1 is None or str2 is None:
        return False

    try:
        ss1 = strip_string(str1)
        ss2 = strip_string(str2)
        if verbose:
            print(ss1, ss2)
        return ss1 == ss2
    except Exception:
        return str1 == str2


def remove_boxed(s):
    if "\\boxed " in s:
        left = "\\boxed "
        assert s[:len(left)] == left
        return s[len(left):]

    left = "\\boxed{"

    assert s[:len(left)] == left
    assert s[-1] == "}"

    return s[len(left):-1]


def last_boxed_only_string(string):
    idx = string.rfind("\\boxed")
    if "\\boxed " in string:
        return "\\boxed " + string.split("\\boxed ")[-1].split("$")[0]
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        retval = None
    else:
        retval = string[idx:right_brace_idx + 1]

    return retval


def fix_fracs(string):
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if substr[0] == "{":
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except AssertionError:
                    return string
                a = substr[0]
                b = substr[1]
                if b != "{":
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}{" + b + "}" + post_substr
                    else:
                        new_str += "{" + a + "}{" + b + "}"
                else:
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}" + b + post_substr
                    else:
                        new_str += "{" + a + "}" + b
    string = new_str
    return string


def fix_a_slash_b(string):
    if len(string.split("/")) != 2:
        return string
    a = string.split("/")[0]
    b = string.split("/")[1]
    try:
        a = int(a)
        b = int(b)
        assert string == "{}/{}".format(a, b)
        new_string = "\\frac{" + str(a) + "}{" + str(b) + "}"
        return new_string
    except AssertionError:
        return string


def remove_right_units(string):
    # "\\text{ " only ever occurs (at least in the val set) when describing units
    if "\\text{ " in string:
        splits = string.split("\\text{ ")
        assert len(splits) == 2
        return splits[0]
    else:
        return string


def fix_sqrt(string):
    if "\\sqrt" not in string:
        return string
    splits = string.split("\\sqrt")
    new_string = splits[0]
    for split in splits[1:]:
        if split[0] != "{":
            a = split[0]
            new_substr = "\\sqrt{" + a + "}" + split[1:]
        else:
            new_substr = "\\sqrt" + split
        new_string += new_substr
    return new_string


def strip_string(string):
    # linebreaks
    string = string.replace("\n", "")

    # remove inverse spaces
    string = string.replace("\\!", "")

    # replace \\ with \
    string = string.replace("\\\\", "\\")

    # replace tfrac and dfrac with frac
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")

    # remove \left and \right
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")

    # Remove circ (degrees)
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")

    # remove dollar signs
    string = string.replace("\\$", "")

    # remove units (on the right)
    string = remove_right_units(string)

    # remove percentage
    string = string.replace("\\%", "")
    string = string.replace("\%", "")  # noqa: W605

    # " 0." equivalent to " ." and "{0." equivalent to "{." Alternatively, add "0" if "." is the start of the string
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")
    # if empty, return empty string
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string

    # to consider: get rid of e.g. "k = " or "q = " at beginning
    if len(string.split("=")) == 2:
        if len(string.split("=")[0]) <= 2:
            string = string.split("=")[1]

    # fix sqrt3 --> sqrt{3}
    string = fix_sqrt(string)

    # remove spaces
    string = string.replace(" ", "")

    # \frac1b or \frac12 --> \frac{1}{b} and \frac{1}{2}, etc. Even works with \frac1{72} (but not \frac{72}1). Also does a/b --> \\frac{a}{b}
    string = fix_fracs(string)

    # manually change 0.5 --> \frac{1}{2}
    if string == "0.5":
        string = "\\frac{1}{2}"

    # NOTE: X/Y changed to \frac{X}{Y} in dataset, but in simple cases fix in case the model output is X/Y
    string = fix_a_slash_b(string)

    return string
