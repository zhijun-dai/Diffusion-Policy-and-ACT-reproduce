#!/bin/bash
# ACT on ALOHA TransferCube — full training
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_ENABLE_HF_TRANSFER=0

lerobot-train \
  --policy.type=act \
  --policy.repo_id=none \
  --policy.push_to_hub=false \
  --env.type=aloha \
  --env.task=AlohaTransferCube-v0 \
  --dataset.repo_id=lerobot/aloha_sim_transfer_cube_human \
  --wandb.enable=false \
  --eval.use_async_envs=false \
  --output_dir=outputs/act_aloha
