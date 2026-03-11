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
FSDP PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import random
import uuid
import torch
import numpy as np

from collections import defaultdict
from tensordict import TensorDict
from pprint import pprint
from tqdm import tqdm
from verl import DataProto
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.core_algos import AdvantageEstimator, agg_loss
from verl.trainer.ppo.core_algos import agg_loss
from verl.trainer.ppo.ray_trainer import RayPPOTrainer, _timer, compute_response_mask
from verl.trainer.ppo.metric_utils import compute_throughout_metrics, compute_timing_metrics, reduce_metrics, compute_data_metrics
import verl.utils.torch_functional as verl_F


def compute_advantage(data: DataProto, adv_estimator: str, config=None) -> DataProto:
    """Compute advantage estimates for policy optimization."""
    # Back-compatible with trainers that do not compute response mask in fit
    if "response_mask" not in data.batch.keys():
        data.batch["response_mask"] = compute_response_mask(data)
    
    if adv_estimator == AdvantageEstimator.RLOO:
        advantages, returns = core_algos.compute_rloo_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=data.batch["response_mask"],
            index=data.non_tensor_batch["uid"],
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    else:
        adv_estimator_fn = core_algos.get_adv_estimator_fn(adv_estimator)
        adv_kwargs = {"token_level_rewards": data.batch["token_level_rewards"],
                      "response_mask": data.batch["response_mask"],
                      "config": config,
        }
        if "uid" in data.non_tensor_batch: # optional
            adv_kwargs['index'] = data.non_tensor_batch["uid"]
        if "reward_baselines" in data.batch:# optional
            adv_kwargs['reward_baselines'] = data.batch["reward_baselines"]

        # calculate advantage estimator
        advantages, returns = adv_estimator_fn(**adv_kwargs)
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    return data


def extract_solution_answer(output: str) -> tuple[str, str]:
    """
    Extract the solution (s) and the final answer (a) from a model's output.

    - s includes all content up to and including the last occurrence of '\\boxed{'.
    - a is the content inside the corresponding braces, excluding the final '}'.
    - Any text following the closing '}' is discarded.
    - Nested braces inside '\\boxed{...}' are correctly handled.

    Args:
        output (str): The full model output containing solution and the final boxed answer.

    Returns:
        tuple[str, str]: A tuple (s, a) where:
            s (str): the solution text up to and including '\\boxed{'.
            a (str): the extracted answer content inside the braces.
    """
    marker = r"\boxed{"
    start = output.rfind(marker)
    if start == -1:
        # No boxed answer found — treat the entire output as solution
        return output, ""

    # Find the matching closing brace, accounting for nested braces
    brace_count = 0
    end = None
    for i in range(start + len(marker), len(output)):
        char = output[i]
        if char == '{':
            brace_count += 1
        elif char == '}':
            if brace_count == 0:
                end = i
                break
            else:
                brace_count -= 1

    # If no closing brace found, assume the string ends after the open box
    if end is None:
        end = len(output)
    s = output[: start + len(marker)]       # include '\boxed{'
    a = output[start + len(marker) : end]   # content inside braces only

    return s, a


def prepare_data_for_reward(data: DataProto, tokenizer, n_samples = 16) -> DataProto:
    """
    Prepare and optionally subsample (question+solution, answer) pairs for reward computing.
    """
    id2answer_ids_list = defaultdict(list)
    id2answer_attention_mask_list = defaultdict(list)
    id2answer_position_list = defaultdict(list)
    input_tensors = defaultdict(list)
    ids = data.non_tensor_batch["uid"]
    dtype = data.batch["prompts"].dtype
    device = data.batch["prompts"].device
    batch_size = len(data)
    
    # Mapping for answer-level subsampling:
    # qid -> answer_str -> sample indices (s_id)
    qid2answer2sid = defaultdict(lambda: defaultdict(list))
    # qid -> answer_str -> answer indices (a_id)
    qid2answer2aid = defaultdict(lambda: defaultdict(list))
    
    for i in range(batch_size):
        data_item = data[i]
        prompt_ids = data_item.batch["prompts"]
        prompt_length = prompt_ids.size(-1)
        valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
        valid_prompt_ids = prompt_ids[-valid_prompt_length:]
        
        response_ids = data_item.batch["responses"]
        response_length = response_ids.size(-1)
        valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
        valid_response_ids = response_ids[:valid_response_length]

        question_str = tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
        response_str = tokenizer.decode(valid_response_ids, skip_special_tokens=True)
        solution_str, answer_str = extract_solution_answer(response_str)
        ground_truth_str = data_item.non_tensor_batch["reward_model"]["ground_truth"]
                
        input = tokenizer(question_str + solution_str, return_tensors="pt", add_special_tokens=False)
        input_ids = input.pop("input_ids").to(device).to(dtype)
        input_attention_mask = input.pop("attention_mask").to(device).to(dtype)
        input_ids, input_attention_mask = verl_F.postprocess_data(
            input_ids=input_ids,
            attention_mask=input_attention_mask,
            max_length=prompt_length+response_length,
            pad_token_id=tokenizer.pad_token_id,
            left_pad=True,
            truncation="right")
        input_position = torch.clip(torch.cumsum(input_attention_mask, dim=-1) - 1, min=0, max=None)
        input_tensors["input_ids"].append(input_ids)
        input_tensors["attention_mask"].append(input_attention_mask)
        input_tensors["position_ids"].append(input_position)
        
        ground_truth = tokenizer(ground_truth_str, return_tensors="pt", add_special_tokens=False)
        ground_truth_ids = ground_truth.pop("input_ids").to(device).to(dtype)
        ground_truth_attention_mask = ground_truth.pop("attention_mask").to(device).to(dtype)
        ground_truth_position = torch.clip(torch.cumsum(ground_truth_attention_mask, dim=-1) - 1, min=0, max=None) + input_position[:, -1] + 1
        if not id2answer_ids_list[ids[i]]:
            id2answer_ids_list[ids[i]].append(ground_truth_ids)
        if not id2answer_attention_mask_list[ids[i]]:
            id2answer_attention_mask_list[ids[i]].append(ground_truth_attention_mask)
        if not id2answer_position_list[ids[i]]:
            id2answer_position_list[ids[i]].append(ground_truth_position)
            
        answer = tokenizer(answer_str, return_tensors="pt", add_special_tokens=False)
        answer_ids = answer.pop("input_ids").to(device).to(dtype)
        answer_attention_mask = answer.pop("attention_mask").to(device).to(dtype)
        answer_position = torch.clip(torch.cumsum(answer_attention_mask, dim=-1) - 1, min=0, max=None) + input_position[:, -1] + 1
        id2answer_ids_list[ids[i]].append(answer_ids)
        id2answer_attention_mask_list[ids[i]].append(answer_attention_mask)
        id2answer_position_list[ids[i]].append(answer_position)
        
        # Record (sample id, answer id) for later subsampling
        answer_id = len(id2answer_ids_list[ids[i]])-1
        qid2answer2sid[ids[i]][answer_str].append(i)
        qid2answer2aid[ids[i]][answer_str].append(answer_id)
        
    for key, val in input_tensors.items():
        input_tensors[key] = torch.cat(val, dim=0)
    
    answer_max_length = max(
        answer_ids.size(-1)
        for answer_ids_list in id2answer_ids_list.values()
        for answer_ids in answer_ids_list
        )
    answer_max_length = min(answer_max_length, 1024)
    unique_ids = list(dict.fromkeys(ids))
    id2answer_ids = defaultdict()
    id2answer_attention_mask = defaultdict()
    id2answer_position = defaultdict()
    for key in unique_ids:
        answer_ids_list = id2answer_ids_list[key]
        answer_attention_mask_list = id2answer_attention_mask_list[key]
        answer_position_list = id2answer_position_list[key]
        for j in range(len(answer_ids_list)):
            answer_ids = answer_ids_list[j]
            answer_attention_mask = answer_attention_mask_list[j]
            answer_position = answer_position_list[j]
            answer_ids, answer_attention_mask = verl_F.postprocess_data(
                input_ids=answer_ids,
                attention_mask=answer_attention_mask,
                max_length=answer_max_length,
                pad_token_id=tokenizer.pad_token_id,
                left_pad=False,
                truncation="right")
            if answer_position.size(-1) < answer_max_length:
                answer_position = verl_F.pad_sequence_to_length(answer_position, max_seq_len=answer_max_length, pad_token_id=0, left_pad=False)
            else:
                answer_position = answer_position[:, :answer_max_length]
            id2answer_ids_list[key][j] = answer_ids
            id2answer_attention_mask_list[key][j] = answer_attention_mask
            id2answer_position_list[key][j] = answer_position
        id2answer_ids[key] = torch.cat(id2answer_ids_list[key], dim=0)
        id2answer_attention_mask[key] = torch.cat(id2answer_attention_mask_list[key], dim=0)
        id2answer_position[key] = torch.cat(id2answer_position_list[key], dim=0)
    
    all_answer_ids = list()
    all_answer_attention_mask = list()
    all_answer_position = list()
    for i in range(batch_size):
        answer_ids = id2answer_ids[ids[i]]
        answer_attention_mask = id2answer_attention_mask[ids[i]]
        answer_position = id2answer_position[ids[i]]
        
        all_answer_ids.append(answer_ids)
        all_answer_attention_mask.append(answer_attention_mask)
        all_answer_position.append(answer_position)
    all_answer_ids = torch.cat(all_answer_ids, dim=0)
    all_answer_attention_mask = torch.cat(all_answer_attention_mask, dim=0)
    all_answer_position = torch.cat(all_answer_position, dim=0)
    
    # the number of generated answer and reference answer for each question
    n_answers = len(list(id2answer_ids_list.values())[0])
    for key, val in input_tensors.items():
        input_tensors[key] = torch.repeat_interleave(val, repeats=n_answers, dim=0)
    
    batch = TensorDict(
        {
            "prompts": input_tensors["input_ids"],
            "responses": all_answer_ids,
            "input_ids": torch.cat([input_tensors["input_ids"], all_answer_ids], dim=-1),
            "attention_mask": torch.cat([input_tensors["attention_mask"], all_answer_attention_mask], dim=-1),
            "position_ids": torch.cat([input_tensors["position_ids"], all_answer_position], dim=-1),
        },
        batch_size=input_tensors["input_ids"].size(0),
    )
    data_for_reward= DataProto(batch=batch, non_tensor_batch={})
    attention_mask_for_reward = all_answer_attention_mask
    full_num_rows = batch_size * n_answers    
    
    # Select global indices to keep
    # global_idx = s_id * n_answers + a_id
    # Subsampling step (always keep ground-truth)
    ground_truth_indices = []
    for i in range(batch_size):
        ground_truth_indices.append(i * n_answers)
    
    # Sample up to n_samples answers per unique answer string
    selected_global_indices = list(ground_truth_indices)
    for qid, answer2sid in qid2answer2sid.items():
        all_s = []
        for v in answer2sid.values():
            all_s.extend(v)
        # select M independent s for every a_id
        # Prioritize selecting those s that can lead to a_id, i.e. sample s from \pi(s|q,a),
        # then randomly sample other s.
        # This can reduce the variance of the estimate.
        for a_str, s_indices in answer2sid.items():
            s_indices = list(dict.fromkeys(s_indices))
            
            if len(s_indices) >= n_samples:
                chosen_s = random.sample(s_indices, n_samples)
            else:
                chosen_s = list(s_indices)
                remaining = n_samples - len(chosen_s)

                candidates = [s for s in all_s if s not in chosen_s]

                if len(candidates) <= remaining:
                    chosen_s.extend(candidates)
                else:
                    chosen_s.extend(random.sample(candidates, remaining))
            for s in chosen_s:
                a_ids = qid2answer2aid[qid][a_str]
                for a_id in a_ids:
                    global_id = s * n_answers + a_id
                    selected_global_indices.append(global_id)

    selected_global_indices = sorted(set(selected_global_indices))
    selected_global_indices_tensor = torch.tensor(
        selected_global_indices,
        device=device,
        dtype=dtype,
    )

    # Build compressed TensorDict using selected indices
    compressed_batch = {}
    for key, val in data_for_reward.batch.items():
        compressed_batch[key] = val.index_select(0, selected_global_indices_tensor)

    compressed_batch = TensorDict(
        compressed_batch,
        batch_size=len(selected_global_indices),
    )

    data_for_reward = DataProto(
        batch=compressed_batch,
        non_tensor_batch={},
    )
    return data_for_reward, attention_mask_for_reward, selected_global_indices_tensor, full_num_rows


def compute_reward_tensor(data_log_prob, attention_mask_for_reward, ids) -> DataProto:
    """
    Compute per-sample reward values from token-level log-probabilities.
    """

    log_prob = data_log_prob.batch["old_log_probs"]
    attention_mask = attention_mask_for_reward
    
    batch_size = len(ids)
    dtype = torch.float32
    device = log_prob.device
    min_val = torch.finfo(dtype).min
    
    masked_log_prob = log_prob * attention_mask
    valid_counts = attention_mask.sum(dim=-1)

    # Sum log-probabilities over valid tokens
    log_prob_seq = masked_log_prob.sum(dim=-1).to(dtype)

    # If a sequence has no valid tokens, assign min_val
    mask_all_zero = (valid_counts == 0)
    log_prob_seq = torch.where(mask_all_zero, torch.tensor(min_val, dtype=dtype, device=device), log_prob_seq)

    log_prob_seq = log_prob_seq.view(batch_size, -1)
    
    # Group sequences by question id
    unique_strs = list(dict.fromkeys(ids.tolist()))
    str2idx = {s: i for i, s in enumerate(unique_strs)}
    ids_int = torch.tensor([str2idx[s] for s in ids], device=device)
    n_groups = len(unique_strs)

    # Boolean mask selecting rows belonging to each group
    mask = (ids_int.unsqueeze(0) == torch.arange(n_groups, device=device).unsqueeze(1))

    # Stack grouped log-probabilities: (n_groups, *, n_answers)
    log_prob_grouped = torch.stack([log_prob_seq[m] for m in mask], dim=0)

    # Compute normalized group-wise reward
    # Reference log-probability (ground-truth)
    log_p0 = log_prob_grouped[:, :, 0]
    # Candidate log-probabilities
    log_pj = log_prob_grouped[:, :, 1:]
    log_norm = torch.logsumexp(log_pj, dim=1, keepdim=True)

    # Extreme min cases
    log_norm_safe = torch.where(log_norm < min_val / 10, torch.zeros_like(log_norm), log_norm)
    log_pj_normalized = log_pj - log_norm_safe
    log_p0_expanded = log_p0.unsqueeze(2).expand_as(log_pj_normalized)
    weight_sum_prob_log = torch.logsumexp(log_p0_expanded + log_pj_normalized, dim=1)
    weight_sum_prob = torch.exp(weight_sum_prob_log)
    
    # Scatter group-wise results back to per-sample tensor
    final_tensor = torch.zeros(batch_size, dtype=dtype, device=device)

    # Preserve original ordering within each group
    group_indices = []
    for g in range(n_groups):
        group_rows = (ids_int == g).nonzero(as_tuple=False).squeeze(1)
        group_rows_sorted = group_rows.sort().values
        group_indices.append(group_rows_sorted)

    scatter_idx = torch.cat(group_indices, dim=0)
    final_tensor[scatter_idx] = weight_sum_prob.flatten()

    return final_tensor


def build_unique_data(data_for_reward):
    """
    Deduplicate identical rows and pad the unique batch
    to be divisible by the number of GPUs.

    Returns:
        unique_data_for_logprob: DataProto with unique (and padded) inputs.
        row_mapping: Tensor mapping each original row to its unique row index.
        unique_len: Number of unique rows before padding.
    """

    batch = data_for_reward.batch
    input_ids = batch["input_ids"]

    # Number of GPUs used for data parallelism
    n_gpus = 8
    
    dtype = input_ids.dtype
    device = input_ids.device

    # Map serialized input_ids -> unique row index
    key2uniq = {}
    # Indices of first occurrences of unique rows
    uniq_row_indices = []
    # For each original row, record its corresponding unique row index
    row_mapping = []

    # Deduplicate rows based on exact input_ids match
    for i in range(input_ids.size(0)):
        key = tuple(input_ids[i].tolist())  # full-sequence equality key
        if key in key2uniq:
            uniq_idx = key2uniq[key]
        else:
            uniq_idx = len(uniq_row_indices)
            key2uniq[key] = uniq_idx
            uniq_row_indices.append(i)
        row_mapping.append(uniq_idx)

    # Convert index lists to tensors
    uniq_row_indices = torch.tensor(uniq_row_indices, dtype=dtype, device=device)
    row_mapping = torch.tensor(row_mapping, dtype=dtype, device=device)

    # Build batch containing only unique rows
    unique_batch = {}
    for k, v in batch.items():
        unique_batch[k] = v.index_select(0, uniq_row_indices)
    
    unique_len = unique_batch["input_ids"].size(0)

    # Pad unique batch to be divisible by n_gpus
    remainder = unique_len % n_gpus
    if remainder != 0:
        pad_num = n_gpus - remainder

        # Reuse the first pad_num rows as padding
        pad_indices = torch.arange(pad_num, device=device)
        pad_unique_batch = {}
        for k, v in unique_batch.items():
            pad_unique_batch[k] = v.index_select(0, pad_indices)

        for k, v in unique_batch.items():
            unique_batch[k] = torch.cat([unique_batch[k], pad_unique_batch[k]], dim=0)

    padded_batch_size = unique_batch["input_ids"].size(0)
    unique_data_for_logprob = DataProto(
        batch=TensorDict(unique_batch, batch_size=padded_batch_size),
        non_tensor_batch={}
    )
    return unique_data_for_logprob, row_mapping, unique_len


def expand_log_prob_to_full(data_log_prob_unique, selected_global_indices, row_mapping, unique_len, full_num_rows):
    device = data_log_prob_unique.batch["old_log_probs"].device
    dtype = data_log_prob_unique.batch["old_log_probs"].dtype
    
    log_prob_unique = data_log_prob_unique.batch["old_log_probs"]
    log_prob_unique = log_prob_unique[:unique_len]

    log_prob_selected = log_prob_unique.index_select(0, row_mapping)

    log_prob_restored = torch.full((full_num_rows, log_prob_unique.size(-1)), torch.finfo(dtype).min, dtype=dtype, device=device)
    
    log_prob_restored[selected_global_indices] = log_prob_selected

    full_batch = TensorDict(
        {"old_log_probs": log_prob_restored},
        batch_size=log_prob_restored.size(0),
    )
    full_data_log_prob = DataProto(batch=full_batch, non_tensor_batch={})
    return full_data_log_prob


class RayCERTrainer(RayPPOTrainer):
    
    def fit(self):
        """
        The training loop of PPO.
        """
        from omegaconf import OmegaConf
        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )
        self.global_steps = 0
        # load checkpoint before doing anything
        self._load_checkpoint()
        
        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        last_val_metrics = None
        
        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                metrics = {}
                timing_raw = {}
                batch: DataProto = DataProto.from_single_dict(batch_dict)

                # pop those keys for generation
                gen_batch = batch.pop(
                    batch_keys=["input_ids", "attention_mask", "position_ids"],
                    non_tensor_batch_keys=["raw_prompt_ids"],
                )

                is_last_step = self.global_steps >= self.total_training_steps

                with _timer("step", timing_raw):
                    # generate a batch
                    with _timer("gen", timing_raw):
                        gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)

                    batch.non_tensor_batch["uid"] = np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object)
                    # repeat to align with repeated responses in rollout
                    batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    batch = batch.union(gen_batch_output)

                    batch.batch["response_mask"] = compute_response_mask(batch)
                    # Balance the number of valid tokens across DP ranks.
                    # NOTE: This usually changes the order of data in the `batch`,
                    # which won't affect the advantage calculation (since it's based on uid),
                    # but might affect the loss calculation (due to the change of mini-batching).
                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                    with _timer("reward", timing_raw):
                        # compute reward model score
                        if self.use_rm:
                            reward_tensor = self.rm_wg.compute_rm_score(batch)
                            batch = batch.union(reward_tensor)
                        
                        data_for_reward, attention_mask_for_reward, selected_global_indices_tensor, full_num_rows = prepare_data_for_reward(batch, self.tokenizer, self.config.algorithm.n_samples)
                        data_for_logprob, row_mapping, unique_len = build_unique_data(data_for_reward)
                        data_log_prob_unique = self.actor_rollout_wg.compute_log_prob(data_for_logprob)
                        data_log_prob_full = expand_log_prob_to_full(data_log_prob_unique, selected_global_indices_tensor, row_mapping, unique_len, full_num_rows)
                        rewards = compute_reward_tensor(data_log_prob_full, attention_mask_for_reward, batch.non_tensor_batch["uid"])
                        metrics.update({"compression/ratio": data_for_logprob.batch["input_ids"].size(0)/full_num_rows})
                        
                        reward_result = self.reward_fn(batch, return_dict=True, rewards=rewards)
                        reward_tensor = reward_result["reward_tensor"]
                        reward_extra_infos_dict = reward_result["reward_extra_info"]
                        
                        if reward_extra_infos_dict:
                            batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})
                                
                    # recompute old_log_probs
                    with _timer("old_log_prob", timing_raw):
                        old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                        entropys = old_log_prob.batch["entropys"]
                        response_masks = batch.batch["response_mask"]
                        loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                        entropy_loss = agg_loss(loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode)
                        old_log_prob_metrics = {"actor/entropy_loss": entropy_loss.detach().item()}
                        metrics.update(old_log_prob_metrics)
                        old_log_prob.batch.pop("entropys")
                        batch = batch.union(old_log_prob)

                    with _timer("adv", timing_raw):
                        batch.batch["token_level_scores"] = reward_tensor
                        batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]
                        batch = compute_advantage(batch, adv_estimator=self.config.algorithm.adv_estimator)

                    # update actor
                    with _timer("update_actor", timing_raw):
                        batch.meta_info["multi_turn"] = self.config.actor_rollout_ref.rollout.multi_turn.enable
                        actor_output = self.actor_rollout_wg.update_actor(batch)
                    actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                    metrics.update(actor_output_metrics)

                    # Log rollout generations if enabled
                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    if rollout_data_dir:
                        with _timer("dump_rollout_generations", timing_raw):
                            print(batch.batch.keys())
                            inputs = self.tokenizer.batch_decode(batch.batch["prompts"], skip_special_tokens=True)
                            outputs = self.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)
                            scores = batch.batch["token_level_scores"].sum(-1).cpu().tolist()
                            self._dump_generations(
                                inputs=inputs,
                                outputs=outputs,
                                scores=scores,
                                reward_extra_infos_dict=reward_extra_infos_dict,
                                dump_path=rollout_data_dir,
                            )

                    # validate
                    if self.val_reward_fn is not None and self.config.trainer.test_freq > 0 and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0):
                        with _timer("testing", timing_raw):
                            val_metrics: dict = self._validate()
                            if is_last_step:
                                last_val_metrics = val_metrics
                        metrics.update(val_metrics)

                    if self.config.trainer.save_freq > 0 and (is_last_step or self.global_steps % self.config.trainer.save_freq == 0):
                        with _timer("save_checkpoint", timing_raw):
                            self._save_checkpoint()

                # training metrics
                metrics.update(
                    {
                        "training/global_step": self.global_steps,
                        "training/epoch": epoch,
                    }
                )
                # collect metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                # TODO: implement actual tflpo and theoretical tflpo
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                progress_bar.update(1)
                self.global_steps += 1
                if is_last_step:
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return
