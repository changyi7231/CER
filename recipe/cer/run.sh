save_dir="/save"

# data
train_files="${save_dir}/data/TIGER-Lab/WebInstruct-verified/train_repeated.parquet"
val_files="['${save_dir}/data/math-ai/math500/test_repeated.parquet', '${save_dir}/data/math-ai/amc23/test_repeated.parquet', '${save_dir}/data/math-ai/aime24/test_repeated.parquet', \
            '${save_dir}/data/math-ai/aime25/test_repeated.parquet', '${save_dir}/data/m-a-p/SuperGPQA/test_repeated.parquet', '${save_dir}/data/TIGER-Lab/MMLU-Pro/test_repeated.parquet']"
max_prompt_length=$((1024 * 2))
max_response_length=$((1024 * 4))  # $((1024 * 8)) for evaluation
train_batch_size=32
filter_overlong_prompts=False
truncation="right"

# actor_rollout_ref
# model
model_path="Qwen/Qwen3-8B-Base"
enable_gradient_checkpointing=True
use_remove_padding=True
# actor
ppo_mini_batch_size=32
use_dynamic_bsz=True
max_token_len_per_gpu=16384
loss_agg_mode="seq-mean-token-sum"
ppo_epochs=1
lr=1e-6
weight_decay=0.0
offload=False
# rollout
temperature=1.0
top_p=1.0
gpu_memory_utilization=0.8
tensor_model_parallel_size=1
rollout_n=16
val_kwargs_temperature=0.6
val_kwargs_top_p=0.95
val_kwargs_top_k=20
val_kwargs_n=1
do_sample=True

# reward_model
reward_manager="cer"

# algorithm
adv_estimator="rloo"
n_samples=16

# trainer
total_epochs=1000000
project_name="CER"
experiment_name=-n_samples${n_samples}
rollout_data_dir="${save_dir}/rollout"
validation_data_dir="${rollout_data_dir}/validation"
nnodes=1
n_gpus_per_node=8
save_freq=50
test_freq=-1
default_local_dir="${save_dir}/model"  # load the checkpoint for evaluation


nohup ray job submit --no-wait \
    -- python -m recipe.cer.src.main_ppo \
    data.train_files="${train_files}" \
    data.val_files="${val_files}" \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.train_batch_size=${train_batch_size} \
    data.filter_overlong_prompts=${filter_overlong_prompts} \
    data.truncation=${truncation} \
    actor_rollout_ref.model.path="${model_path}" \
    actor_rollout_ref.model.enable_gradient_checkpointing=${enable_gradient_checkpointing} \
    actor_rollout_ref.model.use_remove_padding=${use_remove_padding} \
    actor_rollout_ref.actor.ppo_mini_batch_size=${ppo_mini_batch_size} \
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${max_token_len_per_gpu} \
    actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
    actor_rollout_ref.actor.ppo_epochs=${ppo_epochs} \
    actor_rollout_ref.actor.optim.lr=${lr} \
    actor_rollout_ref.actor.optim.weight_decay=${weight_decay} \
    actor_rollout_ref.actor.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${offload} \
    actor_rollout_ref.rollout.temperature=${temperature} \
    actor_rollout_ref.rollout.top_p=${top_p} \
    actor_rollout_ref.rollout.gpu_memory_utilization=${gpu_memory_utilization} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${tensor_model_parallel_size} \
    actor_rollout_ref.rollout.max_num_batched_tokens=${max_token_len_per_gpu} \
    actor_rollout_ref.rollout.n=${rollout_n} \
    actor_rollout_ref.rollout.val_kwargs.temperature=${val_kwargs_temperature} \
    actor_rollout_ref.rollout.val_kwargs.top_p=${val_kwargs_top_p} \
    actor_rollout_ref.rollout.val_kwargs.top_k=${val_kwargs_top_k} \
    actor_rollout_ref.rollout.val_kwargs.n=${val_kwargs_n} \
    actor_rollout_ref.rollout.val_kwargs.do_sample=${do_sample} \
    reward_model.reward_manager=${reward_manager} \
    algorithm.adv_estimator=${adv_estimator} \
    algorithm.n_samples=${n_samples} \
    trainer.total_epochs=${total_epochs} \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${experiment_name}" \
    trainer.rollout_data_dir="${rollout_data_dir}" \
    trainer.validation_data_dir="${validation_data_dir}" \
    trainer.nnodes="${nnodes}" \
    trainer.n_gpus_per_node="${n_gpus_per_node}" \
    trainer.save_freq="${save_freq}" \
    trainer.test_freq="${test_freq}" \
    trainer.default_local_dir="${default_local_dir}" > log.txt 2>&1 &
