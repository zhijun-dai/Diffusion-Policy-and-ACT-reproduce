#!/bin/bash
# Evaluate a trained checkpoint on PushT
export HF_ENDPOINT=https://hf-mirror.com

CKPT=${1:-outputs/diffusion_pusht/checkpoints/100000/pretrained_model}

lerobot-eval \
  --policy.path="$CKPT" \
  --env.type=pusht \
  --eval.n_episodes=50 \
  --eval.use_async_envs=false \
  --output_dir="${CKPT%/*/*}/eval"
