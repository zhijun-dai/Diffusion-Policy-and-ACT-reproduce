#!/bin/bash
# Diffusion Policy on PushT — full training
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_ENABLE_HF_TRANSFER=0

lerobot-train \
  --policy.type=diffusion \
  --policy.repo_id=none \
  --policy.push_to_hub=false \
  --env.type=pusht \
  --dataset.repo_id=lerobot/pusht \
  --wandb.enable=false \
  --eval.use_async_envs=false \
  --output_dir=outputs/diffusion_pusht
