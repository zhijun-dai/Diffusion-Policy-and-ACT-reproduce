# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目目标

复现两篇具身智能论文（基于 HuggingFace LeRobot 框架，纯模拟环境，单 GPU）：

| 论文 | 会议 | 任务 |
|------|------|------|
| Diffusion Policy | RSS 2023 | PushT（2D 推方块） |
| ACT / ALOHA | RSS 2023 | ALOHA 双臂搬运/插入 |

本仓库存放配置文件、评估脚本和实验记录。实际训练代码在 LeRobot 中。

**定位**：项目成果用于向潜在导师展示具身智能方向的兴趣和动手能力。代码和实验记录要经得起审视——不只是跑通，要体现对论文的理解。

## 环境搭建

⚠ 无本地 GPU。训练在云端 GPU 实例上跑（AutoDL 等）。本地只做代码编辑和轻量验证。

```bash
# 云端实例安装 Miniconda
wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
bash /tmp/miniconda.sh -b -p /opt/miniconda && rm /tmp/miniconda.sh

# 创建环境（必须是 Python 3.12，LeRobot 0.6+ 不再支持 3.10）
/opt/miniconda/bin/conda create -y -n lerobot python=3.12

# 安装 LeRobot（所有 extras）
/opt/miniconda/envs/lerobot/bin/pip install -e "/lerobot[pusht,aloha,diffusion,training]"

# 系统依赖（torchcodec 需要）
apt-get install -y -qq ffmpeg
```

## 训练命令

⚠ LeRobot 0.6 命令格式已从 Hydra 风格 (`policy=diffusion`) 改为 argparse 风格 (`--policy.type=diffusion`)。

```bash
# 设置 HF 镜像（国内网络需要）
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_ENABLE_HF_TRANSFER=0

# Diffusion Policy on PushT
lerobot-train \
  --policy.type=diffusion \
  --policy.repo_id=none --policy.push_to_hub=false \
  --env.type=pusht \
  --dataset.repo_id=lerobot/pusht \
  --wandb.enable=false \
  --eval.use_async_envs=false \
  --output_dir=outputs/diffusion_pusht

# ACT on ALOHA TransferCube
lerobot-train \
  --policy.type=act \
  --policy.repo_id=none --policy.push_to_hub=false \
  --env.type=aloha --env.task=AlohaTransferCube-v0 \
  --dataset.repo_id=lerobot/aloha_sim_transfer_cube_human \
  --wandb.enable=false \
  --eval.use_async_envs=false \
  --output_dir=outputs/act_aloha
```

关键参数说明：
- `--eval.use_async_envs=false` 必须加，否则 gym_pusht 等环境在 AsyncVectorEnv 子进程中注册失败
- `--policy.push_to_hub=false --wandb.enable=false` 本地测试时关掉推送
- LeRobot CLI 命令在 `/opt/miniconda/envs/lerobot/bin/`：`lerobot-train`, `lerobot-eval`, `lerobot-record`, `lerobot-replay` 等
- 评估用 `lerobot-eval --policy.path=<checkpoint_path>`

## 参考资源

- LeRobot: https://github.com/huggingface/lerobot
- LeRobot 中文教程: https://github.com/CSCSX/LeRobotTutorial-CN
- Diffusion Policy: https://github.com/real-stanford/diffusion_policy
- ACT: https://github.com/tonyzhaozh/aloha
