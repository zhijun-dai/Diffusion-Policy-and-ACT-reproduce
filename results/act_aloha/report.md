# ACT 复现报告

## 概述

复现论文：**ACT: Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware** (RSS 2023)

基于 LeRobot 框架，在 ALOHA TransferCube 仿真任务上训练 100K 步。

## 实验设置

| 配置 | 值 |
|------|-----|
| 框架 | LeRobot 0.6.1 |
| 环境 | ALOHA TransferCube (双臂搬运方块) |
| 数据集 | lerobot/aloha_sim_transfer_cube_human (50 episodes, 20K frames) |
| GPU | NVIDIA RTX 4090 D 24GB |
| 模型参数量 | 52M |
| Batch size | 8 |
| 训练步数 | 100,000 |
| 内部评估 | 关闭 (env_eval_freq=0) |
| 观测 | 4 cameras (480×640) + 14维关节位置 |
| 输出 | 14维关节位置 |

## 训练过程

| 指标 | Step 200 | Step 20K | Step 60K | Step 100K |
|------|----------|----------|----------|-----------|
| Loss | 7.385 | 0.107 | 0.063 | 0.048 |
| Gradient Norm | 157.9 | 9.9 | 5.6 | 4.3 |
| Learning Rate | 1.0e-5 | 1.0e-5 | 3.1e-5 | 3.3e-10 |

- 训练耗时：~2h20m
- 速度：~12 step/s
- VRAM 占用：2.1 GB

## 评估结果

5 episodes each, 50 evaluation episodes per checkpoint:

| Checkpoint | 成功率 |
|------------|--------|
| 100K | **80.0%** | 5 episodes, 630s eval time |

*ALOHA 评估极慢（400 steps/episode, ~126s/ep），只评估了 100K checkpoint。*

## 与 Diffusion Policy 对比

| 方面 | Diffusion Policy (PushT) | ACT (ALOHA) |
|------|-------------------------|-------------|
| 模型大小 | 263M | 52M |
| VRAM | 5GB | 2GB |
| 训练速度 | ~11.5 step/s | ~12 step/s |
| 任务 | 2D 推方块 | 双臂搬运 |

## 踩坑记录

- MuJoCo headless 需要 `xvfb-run`
- ALOHA 评估极慢，内部评估必须关闭
- 系统盘空间管理：必须用数据盘 (`/root/autodl-tmp`)
- ACT 模型比 Diffusion Policy 小 5 倍，更轻量

## 附录：训练命令

```bash
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_XET=1

lerobot-train \
  --policy.type=act \
  --policy.repo_id=none --policy.push_to_hub=false \
  --env.type=aloha --env.task=AlohaTransferCube-v0 \
  --dataset.repo_id=lerobot/aloha_sim_transfer_cube_human \
  --wandb.enable=false --eval.use_async_envs=false \
  --env_eval_freq=0 \
  --output_dir=outputs/act_aloha
```
