# Diffusion Policy 复现报告

## 概述

复现论文：**Diffusion Policy: Visuomotor Policy Learning via Action Diffusion** (RSS 2023)

基于 HuggingFace LeRobot 框架，在 PushT 任务上训练 100K 步，验证 Diffusion Policy 的核心思想——将机器人动作生成建模为条件扩散去噪过程。

## 实验设置

| 配置 | 值 |
|------|-----|
| 框架 | LeRobot 0.6.1 |
| 环境 | PushT-v0 (gym_pusht) |
| 数据集 | lerobot/pusht (206 episodes, 25,650 frames) |
| GPU | NVIDIA RTX 4090 D 24GB |
| 模型参数量 | 263M |
| Batch size | 8 |
| 训练步数 | 100,000 |
| 图像增强 | 关闭 |
| 观测 | pixels + agent_pos |

### 模型架构

- Vision backbone: ResNet-18 (ImageNet pretrained)
- Diffusion head: U-Net with down_dims=[512, 1024, 2048]
- Noise scheduler: DDPM, 100 timesteps
- Prediction type: epsilon
- Action horizon: 32, observation horizon: 2

## 训练过程

| 指标 | Step 200 | Step 20K | Step 60K | Step 100K |
|------|----------|----------|----------|-----------|
| Loss | 0.390 | 0.027 | 0.018 | 0.010 |
| Gradient Norm | 7.60 | 0.39 | 0.28 | 0.22 |
| Learning Rate | 2.0e-5 | 8.7e-5 | 3.1e-5 | 3.3e-10 |

- 训练耗时：2 小时 31 分
- 速度：~11.5 step/s
- VRAM 占用：4.95 GB

## 评估结果

50 episodes each:

| Checkpoint | 成功率 | Avg Reward | Avg Max Reward |
|------------|--------|------------|----------------|
| 20K | 0% | 53.9 | 0.423 |
| 40K | 6% | 85.2 | 0.687 |
| 60K | 24% | 104.1 | 0.861 |
| 80K | 36% | 100.2 | 0.883 |
| **100K** | **36%** | **105.1** | **0.853** |

## 分析

1. **Loss 下降稳定**：从 0.39 降至 0.01，模型在模仿学习目标上收敛良好
2. **成功率持续提升至 80K 后平稳**：0% → 6% → 24% → 36% → 36%，说明 100K 步接近收敛
3. **与论文差距**：原论文报告 ~70-80% 成功率（200K+ 步，含图像增强），本复现 36%。可能原因：
   - 训练步数不足（论文训练更久）
   - 关闭了图像增强（`image_transforms.enable=False`）
   - 未调参（默认 config）

## 改进方向

- 开启图像随机增强（ColorJitter, RandomAffine 等）
- 延长训练至 200K+ 步
- 调整 diffusion timesteps 和 action horizon
- 尝试 3D Diffusion Policy (DP3) 变体

---

## 附录：环境搭建

```bash
conda create -y -n lerobot python=3.12
pip install -e "/lerobot[pusht,aloha,diffusion,training]"
apt-get install -y ffmpeg
```

## 附录：训练命令

```bash
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_ENABLE_HF_TRANSFER=0

lerobot-train \
  --policy.type=diffusion \
  --policy.repo_id=none --policy.push_to_hub=false \
  --env.type=pusht \
  --dataset.repo_id=lerobot/pusht \
  --wandb.enable=false \
  --eval.use_async_envs=false \
  --output_dir=outputs/diffusion_pusht
```
