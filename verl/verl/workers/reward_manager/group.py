from verl import DataProto
from verl.utils.reward_score import _default_compute_score
import torch


class GroupRewardManager:
    def __init__(
        self,
        tokenizer,
        num_examine,
        compute_score=None,
        rollout_count=1,
        use_token=False,
        use_adaptive_token=False,
        use_anneal_token=False,
        use_target_token=False,
        token_score_weight = 0.025
    ) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.rollout_count = rollout_count
        self.compute_score_fn = (
            compute_score if compute_score is not None else _default_compute_score
        )
        self.use_token = use_token
        self.use_adaptive_token = use_adaptive_token
        self.use_anneal_token = use_anneal_token
        self.use_target_token = use_target_token
        self.token_score_weight =token_score_weight

    def __call__(self, data: DataProto):
        if "rm_scores" in data.batch.keys():
            return data.batch["rm_scores"]

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        acc_tensor =  torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        token_score_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        token_score_weight_tenor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        already_print_data_sources = {}
        record = 0
        for i in range(len(data)):
            try:
                data_item = data[i]
            except:
                breakpoint()
            prompt_ids = data_item.batch["prompts"]

            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch["attention_mask"][
                :prompt_length
            ].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            sequences = torch.cat((valid_prompt_ids, valid_response_ids))
            sequences_str = self.tokenizer.decode(sequences)
            solution_str = self.tokenizer.decode(valid_response_ids)
            ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
            # select rm_score
            data_source = data_item.non_tensor_batch["data_source"]
            rollout_results = []
            for j in range(len(data)):
                if data[j].non_tensor_batch["uid"] == data_item.non_tensor_batch["uid"]:
                    rollout_results.append(data[j])
            score, acc,token_score,token_score_weight = self.compute_score_fn(
                data_source=data_source,
                solution_str=solution_str,
                ground_truth=ground_truth,
                extra_info={
                    "rollout_results": rollout_results,
                    "tokenizer": self.tokenizer,
                    "use_token": self.use_token,
                    "use_adaptive_token": self.use_adaptive_token,
                    "use_anneal_token": self.use_anneal_token,
                    "use_target_token":self.use_target_token,
                    "training_step":data.training_step,
                    "num_steps_per_half_period":data.num_steps_per_half_period,
                    "token_score_weight":self.token_score_weight
                },
            )

            reward_tensor[i, valid_response_length - 1] = score
            acc_tensor[i, valid_response_length - 1] = acc
            token_score_tensor[i, valid_response_length - 1] = token_score
            token_score_weight_tenor[i,valid_response_length-1]  = token_score_weight
            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print(sequences_str)
        return reward_tensor, acc_tensor,token_score_tensor,token_score_weight_tenor
