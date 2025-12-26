#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${REPO_ROOT}"

export PYTHONPATH=src

# Pretrained MDLM checkpoint to load as frozen backbone
MAX_LENGTH="${MAX_LENGTH:-128}"
MDLM_CHECKPOINT="${MDLM_CHECKPOINT:-${REPO_ROOT}/outputs/owt/mdlm_finetune_len${MAX_LENGTH}/outputs/owt/mdlm_finetune_len${MAX_LENGTH}/checkpoints/best.ckpt}"

if [ ! -f "$MDLM_CHECKPOINT" ]; then
    echo "Error: MDLM checkpoint not found at $MDLM_CHECKPOINT"
    echo "Please set MDLM_CHECKPOINT environment variable to point to a valid MDLM checkpoint"
    exit 1
fi

echo "Training Self-Speculative MDLM with backbone from: $MDLM_CHECKPOINT"

python -u -m discrete_diffusion \
    data=openwebtext-split \
    data.wrap=False \
    data.cache_dir=/home/ubuntu/.cache/huggingface/datasets \
    model=self_speculative_dit \
    model.length=${MAX_LENGTH} \
    algo=self_speculative_mdlm \
    sampling=self_speculative \
    training.finetune_path="$MDLM_CHECKPOINT" \
    ++training.strict_load=False \
    training.torch_compile=false \
    loader.batch_size=64 \
    loader.eval_batch_size=64 \
    loader.num_workers=8 \
    trainer.num_nodes=1 \
    trainer.devices=8 \
    trainer.max_steps=50_000 \
    trainer.val_check_interval=1000 \
    trainer.log_every_n_steps=100 \
    callbacks.checkpoint_every_n_steps.every_n_train_steps=5_000 \
    callbacks.checkpoint_every_n_steps.save_top_k=-1 \
    callbacks.checkpoint_every_n_steps.save_last=true \
    callbacks.checkpoint_monitor.save_top_k=1 \
    callbacks.sample_saver.enabled=false \
    checkpointing.resume_from_ckpt=false \
    wandb.project="self_speculative" \
    wandb.name="self_spec_owt_len${MAX_LENGTH}" \
    hydra.run.dir=./outputs/owt/self_speculative_len${MAX_LENGTH}


