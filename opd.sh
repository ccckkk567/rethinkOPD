#!/bin/bash
#SBATCH --job-name=url
#SBATCH --output=logs/20251004/output_%j.log
#SBATCH --error=logs/20251004/error_%j.log
#SBATCH --account=test
#SBATCH --partition=fengl2 
#SBATCH --time=00:20:00
#SBATCH --exclude=g[81-82]
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=500G
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=213231600@seu.edu.cn

# 执行前打印每条命令，便于在日志中定位报错位置。
set -x

# 在非 SBATCH 环境下，额外配置本地日志输出。
if [ -z "$SLURM_JOB_ID" ]; then
    # 为本地运行创建日志目录和日志文件。
    LOG_DIR=${LOG_DIR:-logs}
    mkdir -p "$LOG_DIR"
    LOG_FILE="${LOG_DIR}/run_$(date +%Y%m%d_%H%M%S).log"
    # 同时把输出写到终端和日志文件中。
    exec > >(tee -a "$LOG_FILE") 2>&1
    echo "=========================================="
    echo "Log file: $LOG_FILE"
    echo "Start time: $(date)"
    echo "=========================================="
fi

#================================= 环境配置  ===========================

PROJECT_ROOT="/seu_share2/home/fenglei/213231600/rethinkOPD"
cd "$PROJECT_ROOT"

#mkdir -p logs
#mkdir -p checkpoints

module load anaconda3
source /seu_share/apps/anaconda3/etc/profile.d/conda.sh
export PS1="${PS1:-}"
conda activate verl
export PATH=/seu_share2/home/fenglei/213231600/.conda/envs/verl/bin:$PATH


# -----------------------------------------------------------------------------
# 运行时初始化与分布式调试设置
# -----------------------------------------------------------------------------
# 启动新一轮任务前，先清理可能残留的 Ray 状态。
ray stop --force
export RAY_memory_usage_threshold=0.99
export CUDA_LAUNCH_BLOCKING=1
# export CUDA_VISIBLE_DEVICES=1,2,3,4
export PYTHONUNBUFFERED=1
export PROJECT_NAME='OnPolicyDistillation' # TODO
export TORCH_NCCL_BLOCKING_WAIT=1
export NCCL_TIMEOUT=7200
export TORCH_DISTRIBUTED_DEBUG=INFO

# 目标函数与奖励估计相关设置。
export ADV_ESTIMATOR=token_reward_direct
# export ADV_ESTIMATOR=token_reward_direct_plus_grpo
# export ADV_ESTIMATOR=token_grpo
# export ADV_ESTIMATOR=grpo
export GRPO_OUTCOME_WEIGHT=1.0
# export ADV_ESTIMATOR=token_grpo
# 用于续跑实验的 Swanlab 设置。
# export SWANLAB_RESUME=must
# export SWANLAB_RUN_ID="jri5qia6iy67v7su0zjsv"


# -----------------------------------------------------------------------------
# 长度、采样与损失相关超参数
# -----------------------------------------------------------------------------
# `MAX_MODEL_LEN` 需要覆盖训练和验证两种场景中更大的总序列长度
#（prompt + response）。
export MAX_PROMPT_LENGTH=1024
export MAX_RESP_LENGTH=7168  # TODO: 31744 /15360 / 7168 / 3072 / 5120
export MAX_VAL_RESP_LENGTH=31744 # TODO: 15360 / 7168 / 3072
export MAX_MODEL_LEN=$(( MAX_RESP_LENGTH + MAX_PROMPT_LENGTH > MAX_VAL_RESP_LENGTH + MAX_PROMPT_LENGTH ? MAX_RESP_LENGTH + MAX_PROMPT_LENGTH : MAX_VAL_RESP_LENGTH + MAX_PROMPT_LENGTH ))
export MINI_BATCH_SIZE=${MINI_BATCH_SIZE:-64} # TODO: 1 / 8 / 16 / 32 / 64 (default 64)
export TEMPERATURE=${TEMPERATURE:-1.0} # TODO: 0.6 / 0.8 / 1.0 / 1.2 (default 1.0)
export TEACHER_TEMPERATURE=${TEACHER_TEMPERATURE:-1.0} # 教师 logits 的温度，默认 1.0 表示不缩放
export REPETITION_PENALTY=${REPETITION_PENALTY:-1.0} # TODO: 1.0 / 1.1 / 1.2（默认 1.0，表示不施加重复惩罚）
export N_RESPONSES=4 # TODO: 4 / 8 / 16 / 32（默认建议 8）
export LOG_PROB_TOP_K=${LOG_PROB_TOP_K:-16} # 设为 0 表示不做 top-k 截断
export TOP_K_STRATEGY=${TOP_K_STRATEGY:-"only_stu"} # 可选："only_stu" / "only_tch" / "intersection" / "union" / "union-intersection"
export REWARD_WEIGHT_MODE=${REWARD_WEIGHT_MODE:-"student_p"} # 可选："student_p" / "teacher_p" / "none"
# export LR=${LR:-1e-6}
# export LR_SCHEDULER=${LR_SCHEDULER:-constant}
export USE_KL=${USE_KL:-False} # TODO: True / False（默认 False）
export ENABLE_FORMAT_REWARD=${ENABLE_FORMAT_REWARD:-False} # TODO: True / False（默认 False）
export MODEL_DTYPE=${MODEL_DTYPE:-fp32} # actor/ref/critic 的 fsdp_config.model_dtype，可选 fp32 或 bfloat16
export IS_PLOT=${IS_PLOT:-True} # TODO: True / False（默认 False）
export LOSS_AGG_MODE=${LOSS_AGG_MODE:-"token-mean"} # TODO: "token-mean" / "seq-mean-token-sum" / "seq-mean-token-mean" / "seq-mean-token-sum-norm"（默认 "token-mean"）

# TODO: qwen3_1p7b_base / qwen3_1p7b / llama31_8b_base / llama31_8b_inst / qwen3_8b_base / qwen3_8b / qwen25_1p5b_base / qwen25_1p5b_inst / qwen25_7b_base / qwen25_7b_inst / qwen25_math_7b_base / qwen25_math_7b_inst / qwen25_math_1p5b_base / qwen25_math_1p5b_inst / distill_r1_1p5b / olmo2_1124_7b_base / olmo2_1124_7b_sft / olmo2_1124_7b_inst / llama32_3b_inst
# export EXPERIMENT_NAME=grpo_${TASK}_llama31_tulu3_8b_sft_8k-T_${TEMPERATURE}-n_${N_RESPONSES}-kl_${USE_KL}-mbs_${MINI_BATCH_SIZE}-${REWARD_TYPE}-$(date +%Y-%m-%d_%H-%M-%S)

# -----------------------------------------------------------------------------
# 数据集配置
# -----------------------------------------------------------------------------
# 训练集使用单个 parquet 文件；验证集则以 Hydra 风格的列表字面量传入，
# 这样一次运行可以同时评测多个 benchmark。
# export TRAIN_DATASET=datasets/DAPO-Math-17k/data/dapo-math-17k-10percent.parquet
# export TRAIN_DATASET=datasets/OpenThoughts3-1.2M/OpenThoughts3_opd.parquet
# export TRAIN_DATASET=datasets/OpenThoughts3-1.2M/sampled_complement_30k.parquet
# export TRAIN_DATASET=datasets/DeepMath-103K/verl_format/train_filtered_sampled.parquet
export TRAIN_DATASET=datasets/dapo-math-17k.parquet
# export TRAIN_DATASET=datasets/Skywork-OR1-RL-Data/data/math-00000-of-00001.parquet
# export TRAIN_DATASET=datasets/Skywork-OR1-RL-Data/filtered/math-1p5b-filtered-diff-max8.parquet
# export TRAIN_DATASET=datasets/DAPO-Math-17k-Processed/DAPO-Math.parquet
# export TRAIN_DATASET=datasets/skywork/train_7b_math.parquet
# export TRAIN_DATASET=datasets/DAPO-Math-17k-Processed/DAPO-Math_part2.parquet
# export TRAIN_DATASET=datasets/OpenThoughts3-1.2M/verl_format/train.parquet
export TRAIN_DATASET_NAME=DAPO-Math-17k
# export TRAIN_DATASET_NAME=POLARIS-4B-S1
# export TRAIN_DATASET_NAME=Skywork-OR1-RL-Data
# export TRAIN_DATASET_NAME=DAPO-Math-17k-1percent
# export TRAIN_DATASET_NAME=DeepMath-103K-filtered-sampled
# export TRAIN_DATASET_NAME=DAPO-Math-17k-10percent
# export TRAIN_DATASET_NAME=OpenThoughts3-1.2M-opd
# export TRAIN_DATASET_NAME=OpenThoughts3-1.2M-30k

export TEST_DATA_DIR=datasets/test_data
# TRAIN_DATASET=${TRAIN_FILE:-["$DATA_DIR/$TASK/train_${SAMPLE_SIZE}.parquet"]}
TEST_DATASET=${TEST_FILE:-["$TEST_DATA_DIR/AIME25/test.parquet", "$TEST_DATA_DIR/AMC23/test.parquet", "$TEST_DATA_DIR/AIME24/test.parquet"]}
# TEST_DATASET=${TEST_FILE:-["$TEST_DATA_DIR/AIME24/test.parquet"]}
# TEST_DATASET=${TEST_FILE:-["$DATA_DIR/AIME24/test.parquet","$DATA_DIR/AIME25/test.parquet","$DATA_DIR/AMC23/test.parquet","$DATA_DIR/MATH-500/test.parquet","$DATA_DIR/Minerva/test.parquet","$DATA_DIR/Olympiad-Bench/test.parquet"]}

# -----------------------------------------------------------------------------
# 模型选择
# -----------------------------------------------------------------------------
# 用 `basename` 提取模型目录名，便于实验名和 checkpoint 路径更简洁。
# TODO:
# export ACTOR_MODEL_PATH=model/qwen3-1.7b-math-sft
# export ACTOR_MODEL_PATH=model/DS-1.5B-sft
# export ACTOR_MODEL_PATH=model/DS-1.5B-sft-skywork
# export ACTOR_MODEL_PATH=model/DS-1.5B-sft-ds-7b
# export ACTOR_MODEL_PATH=/workspace/model/Qwen3-1.7B-SFT-DAPO-4B-RL
# export ACTOR_MODEL_PATH=/workspace/model/Qwen3-1.7B-SFT-DAPO-4B
# export ACTOR_MODEL_PATH=model/Qwen2.5-Math-1.5B
export ACTOR_MODEL_PATH=model/DeepSeek-R1-Distill-Qwen-1.5B
# export ACTOR_MODEL_PATH=model/JustRL-DeepSeek-1.5B-step_0400
# export ACTOR_MODEL_PATH=model/JustRL-DeepSeek-1.5B
# export ACTOR_MODEL_PATH=model/Qwen3-1.7B-SFT
# export ACTOR_MODEL_PATH=model/Qwen3-1.7B-Base-SFT-OpenThought3-4B/checkpoint-1800
# export ACTOR_MODEL_PATH=model/Qwen3-1.7B-Base
# export ACTOR_MODEL_PATH=model/Qwen3-1.7B
# export ACTOR_MODEL_PATH=model/Qwen3-1.7B-Base-SFT-DeepMath-4B
# export ACTOR_MODEL_PATH=model/Qwen3-1.7B-sft/checkpoint-6000
# export ACTOR_MODEL_PATH=model/DeepSeek-R1-Distill-Qwen-7B
# export ACTOR_MODEL_PATH=model/DS-1.5B-SFT
export ACTOR_MODEL_NAME=$(basename "$ACTOR_MODEL_PATH")
# export REWARD_MODEL_PATH=model/Qwen3-4B
# export REWARD_MODEL_PATH=model/Qwen3-4B-grpo
# export REWARD_MODEL_PATH=model/Qwen3-1.7B
# export REWARD_MODEL_PATH=model/OpenMath-Nemotron-1.5B
# export REWARD_MODEL_PATH=model/DeepSeek-R1-Distill-Qwen-7B
# export REWARD_MODEL_PATH=model/Qwen3-4B-Non-Thinking-RL-Math
# export REWARD_MODEL_PATH=model/Skywork-OR1-Math-7B
# export REWARD_MODEL_PATH=model/Polaris-4B-Preview
# export REWARD_MODEL_PATH=model/DeepSeek-R1-Distill-Qwen-14B
export REWARD_MODEL_PATH=model/JustRL-DeepSeek-1.5B
export REWARD_MODEL_NAME=$(basename "$REWARD_MODEL_PATH")

# -----------------------------------------------------------------------------
# 运行时派生出的元信息与缓存目录
# -----------------------------------------------------------------------------
# 在运行名中加入时间戳，避免多次启动时发生重名覆盖。
export PROJECT_PATH=checkpoint
export PARALLEL_SIZE=1
export CKPT_PATH=${PROJECT_PATH}/${ADV_ESTIMATOR}_${TRAIN_DATASET_NAME}_${ACTOR_MODEL_NAME}_${REWARD_MODEL_NAME}_${MAX_RESP_LENGTH}-T_${TEMPERATURE}-Tch_${TEACHER_TEMPERATURE}-n_${N_RESPONSES}-mbs_${MINI_BATCH_SIZE}-topk_${LOG_PROB_TOP_K}-topk_strategy_${TOP_K_STRATEGY}-rw_${REWARD_WEIGHT_MODE}-$(date +%Y-%m-%d_%H-%M-%S)
export OUTLINES_CACHE_DIR=~/.cache/outlines/$(uuidgen)
export NCCL_DEBUG=WARN

# export VLLM_ATTENTION_BACKEND=XFORMERS
# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=true
export SWANLAB_LOG_DIR=${PROJECT_PATH}/swanlab_log
export HYDRA_FULL_ERROR=1


export EXPERIMENT_NAME=${ADV_ESTIMATOR}_${TRAIN_DATASET_NAME}_${ACTOR_MODEL_NAME}_${REWARD_MODEL_NAME}_${MAX_RESP_LENGTH}-T_${TEMPERATURE}-Tch_${TEACHER_TEMPERATURE}-n_${N_RESPONSES}-mbs_${MINI_BATCH_SIZE}-topk_${LOG_PROB_TOP_K}-topk_strategy_${TOP_K_STRATEGY}-rw_${REWARD_WEIGHT_MODE}-$(date +%Y-%m-%d_%H-%M-%S)

# -----------------------------------------------------------------------------
# 可选的 Hydra 参数片段
# -----------------------------------------------------------------------------
# 按条件拼接命令行参数片段，让主启动命令更易读。
KL_ARGS=""
if [ "$USE_KL" = "True" ]; then
    KL_ARGS="actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.005 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl"
else
    KL_ARGS="actor_rollout_ref.actor.use_kl_loss=False"
fi

LR_ARGS=""
if [ "$LR_SCHEDULER" = "cosine" ]; then
    LR_ARGS="actor_rollout_ref.actor.optim.warmup_style=cosine \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.03"
fi

# 为每张 GPU 计算 PPO 的 token 上限，至少保证不小于 32768，
# 同时也要容纳 prompt + response 的总长度。
PPO_MAX_TOKEN_LEN_PER_GPU=$(( ((1024 + MAX_RESP_LENGTH) > 32768) ? (1024 + MAX_RESP_LENGTH) : 32768))
echo "PPO_MAX_TOKEN_LEN_PER_GPU: $PPO_MAX_TOKEN_LEN_PER_GPU"


# 为 VERL worker 启动本地 Ray head 节点。
ray start --head
sleep 5


# 通过 Hydra override 的方式启动 VERL PPO 训练。
# 带 `+` 前缀的键表示向基础配置中追加原本可能不存在的字段。
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=$ADV_ESTIMATOR \
    algorithm.grpo_outcome_weight=$GRPO_OUTCOME_WEIGHT \
    data.shuffle=False \
    data.train_files="$TRAIN_DATASET" \
    data.val_files="$TEST_DATASET" \
    data.train_batch_size=$((${MINI_BATCH_SIZE}*${PARALLEL_SIZE})) \
    data.max_prompt_length=$MAX_PROMPT_LENGTH \
    data.max_response_length=$MAX_RESP_LENGTH \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=$ACTOR_MODEL_PATH \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_activation_offload=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    $LR_ARGS \
    actor_rollout_ref.actor.ppo_mini_batch_size=$MINI_BATCH_SIZE \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$PPO_MAX_TOKEN_LEN_PER_GPU \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=$PARALLEL_SIZE \
    $KL_ARGS \
    actor_rollout_ref.actor.loss_agg_mode=$LOSS_AGG_MODE \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.fsdp_config.forward_prefetch=True \
    actor_rollout_ref.actor.fsdp_config.model_dtype=$MODEL_DTYPE \
    actor_rollout_ref.rollout.max_num_batched_tokens=$PPO_MAX_TOKEN_LEN_PER_GPU \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.fsdp_config.model_dtype=$MODEL_DTYPE \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.temperature=$TEMPERATURE \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    +actor_rollout_ref.rollout.log_prob_top_k=$LOG_PROB_TOP_K \
    +actor_rollout_ref.rollout.top_k_strategy=$TOP_K_STRATEGY \
    +actor_rollout_ref.rollout.reward_weight_mode=$REWARD_WEIGHT_MODE \
    +actor_rollout_ref.rollout.teacher_temperature=$TEACHER_TEMPERATURE \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$PARALLEL_SIZE \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.max_model_len=$MAX_MODEL_LEN \
    actor_rollout_ref.rollout.n=$N_RESPONSES \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    +actor_rollout_ref.rollout.val_kwargs.max_tokens=$MAX_VAL_RESP_LENGTH \
    actor_rollout_ref.rollout.val_kwargs.n=16 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.7 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
    actor_rollout_ref.rollout.repetition_penalty=$REPETITION_PENALTY \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    reward_model.enable=True \
    +reward_model.reward_kwargs.enable_format_reward=$ENABLE_FORMAT_REWARD \
    reward_model.model.path=$REWARD_MODEL_PATH \
    reward_model.model.input_tokenizer=null \
    reward_model.model.use_remove_padding=True \
    reward_model.model.fsdp_config.param_offload=False \
    +reward_model.model.dtype=$MODEL_DTYPE \
    reward_model.micro_batch_size_per_gpu=24 \
    custom_reward_function.path="verl/verl/utils/reward_score/ttrl_math/__init__.py" \
    custom_reward_function.name=reward_func \
    trainer.val_before_train=False \
    trainer.log_val_generations=2 \
    trainer.logger=['console','swanlab'] \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.validation_data_dir=validation_log/$EXPERIMENT_NAME \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    trainer.test_freq=20 \
    trainer.total_epochs=1 \
    trainer.default_local_dir="$CKPT_PATH" \
    trainer.is_plot=$IS_PLOT \

# 本地运行时额外记录结束时间。
if [ -z "$SLURM_JOB_ID" ]; then
    echo "=========================================="
    echo "End time: $(date)"
    echo "=========================================="
fi
