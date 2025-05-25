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
# from . import gsm8k, math, prime_math, prime_code


def _default_compute_score(data_source,
                           solution_str,
                           ground_truth,
                           extra_info=None):
    if data_source == 'openai/gsm8k':
        from . import gsm8k
        res = gsm8k.compute_score(solution_str, ground_truth)
    elif data_source in [
            'lighteval/MATH', 'DigitalLearningGmbH/MATH-lighteval',"deepsacler_validation"
    ]:
        from . import math
        res = math.compute_score(solution_str, ground_truth)
        return res
    elif data_source in [
            'numina_aops_forum', 'numina_synthetic_math', 'numina_amc_aime',
            'numina_synthetic_amc', 'numina_cn_k12', 'numina_olympiads'
    ]:
        from . import prime_math
        res = prime_math.compute_score(solution_str, ground_truth)
    elif data_source in ['codecontests', 'apps', 'codeforces', 'taco']:
        from . import prime_code
        res = prime_code.compute_score(solution_str,
                                       ground_truth,
                                       continuous=True)
    elif data_source in ["deepsacler_selective_token"]:
        from . import math_select
        res = math_select.compute_score(
            solution_str=solution_str,
            ground_truth=ground_truth,
            rollout_results=extra_info["rollout_results"],
            tokenizer=extra_info["tokenizer"],
            use_token= extra_info["use_token"],
            use_adaptive_token = extra_info["use_adaptive_token"],
            use_anneal_token= extra_info["use_anneal_token"],
            training_step=extra_info["training_step"],
            num_steps_per_half_period=extra_info["num_steps_per_half_period"],
            use_target_token=extra_info["use_target_token"],
            token_score_weight=extra_info["token_score_weight"]
        )
        return res
    else:
        raise NotImplementedError

    if isinstance(res, (int, float, bool)):
        return float(res)
    else:
        return float(res[0])
