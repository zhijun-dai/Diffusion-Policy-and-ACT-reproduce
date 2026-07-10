# 具身智能论文复现

西安交通大学 人工智能专业 大一升大二，复现具身智能领域经典论文，作为联系导师的证明材料。

## 个人背景

- 已掌握基础：Transformer (Attention Is All You Need)、ResNet、GAN（见 `D:\用户\Lenovo\桌面\论文\经典\`）
- 已读过 CLIP，了解视觉-语言对齐基本范式
- 具身智能已读入门综述：
  - Aligning Cyber Space with Physical World: A Comprehensive Survey on Embodied AI
  - A Survey on Vision-Language-Action Models for Embodied AI
- 必读论文清单：`D:\用户\Lenovo\桌面\论文\必读论文\`
- 文献管理：Zotero 9，默认存储模式（PDF 在 Zotero data 目录下）

## 复现目标

| 论文 | 会议 | 任务 | 难度 |
|------|------|------|------|
| Diffusion Policy: Visuomotor Policy Learning via Action Diffusion | RSS 2023 | PushT（2D 模拟，推方块到位） | ★★☆ |
| ACT: Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware | RSS 2023 | ALOHA 模拟（双臂搬运/插入） | ★★★ |

两篇均通过 HuggingFace LeRobot 框架复现，纯模拟环境，单 GPU 可跑。

## 环境搭建

```bash
git clone https://github.com/huggingface/lerobot.git
cd lerobot
conda create -y -n lerobot python=3.12
conda activate lerobot
pip install -e ".[pusht, aloha, diffusion, training]"
# torchcodec 需要系统 ffmpeg
sudo apt-get install -y ffmpeg
```

## 训练命令

> LeRobot 0.6+ 命令格式已从 `policy=diffusion` (Hydra) 改为 `--policy.type=diffusion` (argparse)。

```bash
# 国内网络配置 HF 镜像
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

## 参考资源

- LeRobot 中文教程：https://github.com/CSCSX/LeRobotTutorial-CN
- LeRobot 官方文档：https://github.com/huggingface/lerobot
- Diffusion Policy 官方仓库：https://github.com/real-stanford/diffusion_policy
- ACT 官方仓库：https://github.com/tonyzhaozh/aloha
- 具身 VLA 论文清单：https://github.com/jonyzhang2023/awesome-embodied-vla-va-vln
