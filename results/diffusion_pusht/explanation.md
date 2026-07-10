# Diffusion Policy & ACT 复现：完整讲解

> 这份文档帮你理解：原论文讲了什么、代码怎么跑的、两个模型有什么区别、面聊时怎么回答。

---

## 零、预备知识：DDPM 扩散模型是怎么工作的

在理解 Diffusion Policy 之前，先搞清楚基础的扩散模型（DDPM, Ho et al. 2020）。

### 0.1 前向过程：逐步加噪

有一张干净图片（或一条动作轨迹）x_0。我们往上面逐步加高斯噪声：

```
x_0 (干净) → x_1 (加一点噪声) → x_2 (更多噪声) → ... → x_T (纯噪声)
```

每一步：`x_t = sqrt(1-β_t) * x_{t-1} + sqrt(β_t) * ε`，其中 ε ~ N(0,I)，β_t 是噪声调度。

关键性质：可以直接从 x_0 跳到任意步 x_t，不需要一步步算：
```
x_t = sqrt(ᾱ_t) * x_0 + sqrt(1-ᾱ_t) * ε
```
其中 α_t = 1-β_t, ᾱ_t = α_1 * α_2 * ... * α_t。

### 0.2 反向过程：逐步去噪

如果能从 x_t 恢复出 x_{t-1}，就能从纯噪声一步步恢复出干净数据。

DDPM 的核心发现：**不需要直接预测 x_{t-1}，只需要预测加入的噪声 ε**。

训练目标：给模型看 (x_t, t)，让它预测 ε。Loss = MSE(预测噪声, 真实噪声)。

```
训练: x_0 → 加随机噪声 ε → 得到 x_t → 模型预测 ε̂ → Loss = |ε̂ - ε|²
推理: x_T ~ N(0,I) → 模型预测 ε̂ → 去噪一步得 x_{T-1} → ... → x_0
```

### 0.3 条件扩散

在去噪过程中加入条件 c（比如"一只猫"、"当前观测画面"），引导生成方向。

就是把条件拼入每一步的输入：`ε̂ = model(x_t, t, c)`。

这正是 Diffusion Policy 做的事情：**条件 c = 图像观测 + 机器人状态**。

---

## 一、Diffusion Policy 原论文解析

### 1.1 论文基本信息

- 标题：Diffusion Policy: Visuomotor Policy Learning via Action Diffusion
- 作者：Cheng Chi et al. (Columbia, Stanford)
- 会议：RSS 2023 (Robotics: Science and Systems)
- 代码：github.com/real-stanford/diffusion_policy

### 1.2 核心创新点

**创新 1：把动作生成变成条件扩散**

传统模仿学习：神经网络直接输出动作坐标。Diffusion Policy：给观测条件，用扩散模型生成动作轨迹。这不是简单的"换了个网络结构"，而是根本上改变了动作分布的建模方式。

**创新 2：动作序列级别的去噪（Action Sequence Denoising）**

不是每步去噪一个动作，而是对整条动作轨迹（horizon=64 步）一起去噪。这样模型能学到动作之间的时序连贯性——不会出现相邻两步动作方向相反的奇怪情况。

**创新 3：receding horizon 控制**

预测 64 步，但只执行前 32 步。32 步后再预测下一段。这就像 MPC（模型预测控制）的思路——随时根据最新观测重新规划。

```
时间线:  |---预测64步---|
         |--执行32步--||---重新预测64步---|
                                  |--执行32步--|...|
```

**创新 4：用 U-Net 1D 而不是 Transformer**

为什么不用 Transformer？因为动作轨迹在时间维度上空间结构很强（相邻步之间变化平滑），1D 卷积天然适合捕捉局部时序模式。而 Transformer 虽然全局感受野大，但计算量高且需要更多数据。

### 1.3 与基线方法的对比

| 方法 | 做法 | 问题 |
|------|------|------|
| **Behavior Cloning (BC)** | CNN → MLP 直接回归动作 | 多模态分布坍缩成均值 |
| **LSTM-GMM** | LSTM 预测高斯混合模型参数 | 混合数固定,表达能力有限 |
| **IBC (Implicit BC)** | 能量模型,需采样推断 | 推理慢,训练不稳定 |
| **Diffusion Policy** | 扩散模型去噪生成动作 | 表达能力强,训练稳定但推理稍慢 |

论文的实验结论：Diffusion Policy 在所有 12 项测试任务上比 BC 高出 20-50% 成功率。

### 1.4 PushT 任务

PushT 是 Diffusion Policy 论文的标准测试环境。

- 桌上有一个 T 形方块（形状不对称）
- 目标：把方块推到指定位置和角度（位置 + 旋转共 3 自由度）
- 输入：俯视 RGB 图像 (384×384) + 末端执行器位置 (2维 x,y)
- 输出：末端执行器位移 (2维)
- 难度在于需要精确控制方块的朝向——推偏一点就歪了

---

## 二、训练流程全景

### 2.1 数据流 + Tensor Shape 追踪

这是训练中每个阶段的 tensor 形状变化，看清数据怎么流转的：

```
DataLoader 输出一个 batch (B=8):
├── observation.state      (8, 2, 2)      [B, n_obs_steps, state_dim]
├── observation.image       (8, 2, 3, 384, 384)  [B, n_obs_steps, C, H, W]
└── action                  (8, 64, 2)     [B, horizon, action_dim]

进入 compute_loss():
│
├── _prepare_global_conditioning():
│   ├── ResNet-18(image) → (8, 2, 512, 12, 12) 特征图
│   │   重组为 (16, 512, 12, 12) [合并B和S维度]
│   ├── SpatialSoftmax → (16, 32, 2) 32个关键点坐标
│   ├── Flatten → (16, 64) 图像特征
│   ├── 恢复为 (8, 2, 64) [分离B和S]
│   ├── state: (8, 2, 2)
│   ├── 拼接 → (8, 2, 66)  [state_dim + img_feat_dim = 2 + 64]
│   └── Flatten → (8, 132)  [2 × 66]  = global_cond
│
├── 加噪:
│   ε = randn(8, 64, 2)
│   timestep = randint(0, 100) for each sample → (8,)
│   noisy = scheduler.add_noise(action, ε, timestep) → (8, 64, 2)
│
├── U-Net forward:
│   noisy (8, 2, 64) [rearrange: B D T] + timestep (8,) + global_cond (8, 132)
│   ↓ down_modules [512 → 1024 → 2048, 逐步降采样]
│   ↓ mid_modules [2048 → 2048]
│   ↓ up_modules [2048 → 1024 → 512, skip connections]
│   ↓ final_conv [512 → 2] 输出
│   → pred ε̂ (8, 64, 2) [rearrange back: B T D]
│
└── Loss = MSE(ε̂, ε)
```

### 2.2 训练循环 (`lerobot_train.py`)

```python
for batch in dataloader:
    loss, _ = policy.forward(batch)             # 前向 + loss
    accelerator.backward(loss)                  # 反向传播
    clip_grad_norm_(policy.parameters(), 10.0)   # 梯度裁剪
    optimizer.step()                             # 更新参数
    lr_scheduler.step()                          # 调整学习率
```

### 2.3 Loss 计算源码注释版

```python
def compute_loss(self, batch):
    # === 1. 编码视觉特征 ===
    # ResNet-18 → SpatialSoftmax → 64维/图 + 2维状态 = 66维
    # × 2步观测 = 132维条件向量
    global_cond = self._prepare_global_conditioning(batch)
    
    # === 2. 前向扩散：给动作加噪声 ===
    trajectory = batch["action"]           # (B, 64, 2)
    eps = torch.randn_like(trajectory)     # 随机噪声
    # 每个样本随机选一个去噪时刻（0=无噪声, 99=几乎纯噪声）
    timesteps = torch.randint(0, 100, (B,)).long()
    # 根据时刻加对应幅度的噪声
    noisy = self.noise_scheduler.add_noise(trajectory, eps, timesteps)
    
    # === 3. U-Net 去噪 ===
    # 输入: 加噪动作 + 时间步 + 观测条件
    # 输出: 预测的噪声
    pred = self.unet(noisy, timesteps, global_cond=global_cond)
    
    # === 4. 损失：预测噪声 vs 真实噪声 ===
    loss = F.mse_loss(pred, eps)
    return loss
```

---

## 三、模型架构详解

### 3.1 整体结构

```
DiffusionPolicy (263M params)
└── DiffusionModel
    ├── rgb_encoder: ResNet-18 (去掉最后两层)
    │   ├── SpatialSoftmax → 32个关键点坐标
    │   └── Linear(64→64) + ReLU → 64维特征
    │
    ├── unet: DiffusionConditionalUnet1d
    │   ├── down_modules: [512, 1024, 2048] 编码器
    │   │   每个block: Conv1d → GroupNorm → Mish, ×2 + downsample
    │   ├── mid_modules: [2048, 2048] 瓶颈
    │   ├── up_modules: [1024, 512] 解码器 (skip connections)
    │   └── final_conv: Conv1d(512→128→2)
    │
    └── noise_scheduler: DDPMScheduler (T=100步, β: linear schedule)
```

### 3.2 视觉编码器：SpatialSoftmax

不同于普通 CNN 最后做全局平均池化取一个标量，SpatialSoftmax 做的是：

1. 特征图每个通道 (512, 12, 12) 做 softmax 归一化
2. 算每个通道激活的加权质心 → 得到 (512, 2) 的坐标
3. 用 1×1 conv 压到 32 通道 → (32, 2) 个关键点

**好处**：关键点在图像空间的坐标具有物理含义——它们可能对应 T 形方块的角点、末端执行器的位置等。这比直接 flatten 做 MLP 更有空间感知能力。

### 3.3 1D U-Net 的 FiLM 条件注入

FiLM = Feature-wise Linear Modulation。在做卷积之前，用条件向量调整特征图的偏置（和可选的缩放）：

```python
cond_embed = Linear(Mish(Linear(cond)))   # cond_dim → out_channels
out = conv(x) + cond_embed[:, :, None]    # broadcast 到每个时间步
```

这等价于在每个卷积层告诉网络"这些观测下，你应该去噪到这个方向"。

### 3.4 推理过程

```python
# 1. 从纯噪声开始
action = torch.randn(B, 64, 2)            # 64步2D位移的随机值

# 2. 逐步去噪（100步）
for t in range(99, -1, -1):
    noise_pred = unet(action, t, global_cond)
    action = scheduler.step(noise_pred, t, action)
    # 每步去掉 predictions 中的一部分噪声

# 3. 取前32步执行
return action[:, 31:63]  # 从第n_obs_steps-1位开始取
```

DDIM 加速：如果不想跑满 100 步，可以用 DDIM scheduler 跳到只用 10-20 步。论文实验表明 10 步 DDIM 几乎不影响成功率。

---

## 四、我们的实验配置

### 4.1 训练参数

| 参数 | 值 | 含义 |
|------|-----|------|
| batch_size | 8 | 每批样本数 |
| steps | 100,000 | 总训练步数 |
| n_obs_steps | 2 | 用最近 2 帧观测 |
| horizon | 64 | 预测未来 64 步动作 |
| n_action_steps | 32 | 实际执行前 32 步 |
| num_train_timesteps | 100 | 扩散步数 |
| optimizer | Adam (β=[0.95, 0.999]) | 优化器 |
| lr | 1e-4 → 0 (cosine warmup 500) | 余弦退火 |
| vision_backbone | resnet18 (ImageNet预训练) | 图像编码器 |
| down_dims | [512, 1024, 2048] | U-Net通道 |
| noise_scheduler | DDPM | β 从 1e-4 到 0.02 |

### 4.2 数据细节

- PushT 数据集：206 episodes, 25,650 frames
- 每 episode 约 125 帧 (12.5 秒 @ 10fps)
- 训练/评估无 split（全量训练），因为我们只做 offline BC

---

## 五、结果解读

### 5.1 训练曲线

| Step | Loss | Grad Norm | LR | 说明 |
|------|------|-----------|-----|------|
| 200 | 0.390 | 7.60 | 2.0e-5 | 初始，去噪能力差 |
| 20K | 0.027 | 0.39 | 8.7e-5 | 学会基本的去噪 |
| 60K | 0.018 | 0.28 | 3.1e-5 | 持续收敛 |
| 100K | 0.010 | 0.22 | 3.3e-10 | 接近收敛 |

### 5.2 成功率

| Checkpoint | 成功率 | 解读 |
|------------|--------|------|
| 20K | 0% | loss低但实际不会推（过拟合？） |
| 40K | 6% | 开始有成功的 |
| 60K | 24% | 跳跃提升 |
| 80K | 36% | 趋于收敛 |
| 100K | 36% | 平稳 |

**为什么 loss 在降但成功率不涨？** 因为 loss 衡量的是"去噪精度"，不完全等价于"任务成功率"。模型可能在数据集的常见场景下去噪很准，但遇到评估时的 unseen 初始状态就表现差。这也说明了 offline BC 的核心问题：分布偏移 (distribution shift)。

### 5.3 VRAM

- 训练: ~5GB / 24GB
- 模型 263M 参数
- 预估在任意 >6GB 显存的 GPU 上都能跑

---

## 六、ACT (Action Chunking Transformer) 详解

### 6.1 论文基本信息

- 标题：Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware
- 作者：Tony Z. Zhao et al. (Stanford)
- 会议：RSS 2023
- 特点：论文主要贡献之一是 ALOHA 低成本双臂遥操作硬件（$32K vs 传统>$100K）

### 6.2 核心思想

ACT 的核心可以用三句话概括：

**CVAE (Conditional VAE)**：不是直接学 observation→action 的映射，而是学 observation + latent z → action。z 编码了动作的"风格"或"意图"。训练时 z 从后验分布 q(z|obs, action) 采样，推理时从先验 N(0,I) 采样。

**Action Chunking**：一次预测一整段动作序列（chunk），而不是逐帧预测。这样相邻动作之间保持时序一致性。

**Temporal Ensemble**：因为用 receding horizon 控制，每一步会被多次预测（不同 chunk 的重叠部分）。对多次预测取指数加权平均，让动作更平滑。

### 6.3 架构

```
观测 (图像 + 关节位置)
  │
  ├── ResNet-18 → 图像特征
  ├── concat with joint positions
  │
  ▼
Transformer Encoder (4层) → 编码观测
  │
  ├── 后验: Transformer Decoder (obs + action → z_mean, z_logvar)
  │   训练时：z ~ q(z|obs, action)
  │   推理时：z ~ N(0, I)
  │
  ▼
Transformer Decoder → 动作序列 (chunk_size × action_dim)
```

### 6.4 与 Diffusion Policy 的对比

| 方面 | Diffusion Policy | ACT |
|------|-----------------|-----|
| **生成方式** | 迭代去噪 (100步) | 一次前向 |
| **推理速度** | 较慢 (≈0.1s, 可DDIM加速) | 很快 (≈5ms) |
| **动作多样性** | 高（扩散天然支持多模态） | 中（VAE latent） |
| **时序一致性** | U-Net 1D卷积保证 | Chunking + Temporal Ensemble |
| **模型大小** | 263M | 52M |
| **适用场景** | 高精度、多模态任务 | 高速、实时控制 |
| **训练稳定性** | 很稳定 | 需要调 β (KL权重) |

### 6.5 我们的 ACT 实验配置

| 参数 | 值 |
|------|-----|
| 环境 | ALOHA TransferCube (双臂搬运方块) |
| 数据集 | lerobot/aloha_sim_transfer_cube_human (50 eps) |
| 模型参数量 | 52M |
| chunk_size | 100 |
| 输入 | 4个相机 (640×480) + 14维关节位置 |
| 输出 | 14维关节位置 |

---

## 七、调参技巧与踩坑记录

### 7.1 Diffusion Policy 调参

| 参数 | 效果 | 建议 |
|------|------|------|
| n_obs_steps | 更多=更多历史信息 | PushT 用2够了,动态任务用4-8 |
| horizon | 更长=看得更远 | 太长增加计算,太短缺乏规划 |
| num_train_timesteps | 更多=扩散质量更好 | 100是性价比甜点 |
| image_transforms | 防止过拟合 | **开启后预期成功率+20%** |
| lr | 太高不收敛太低慢 | 1e-4 → cosine decay 通用 |

### 7.2 ACT 调参

| 参数 | 效果 | 建议 |
|------|------|------|
| kl_weight (β) | 控制 latent 约束强度 | 太小=VAE退化,太大=忽略latent |
| chunk_size | 一次预测的长度 | 太短=动作不连贯,太长=难学 |
| Temporal Ensemble | 平滑多步预测 | 推断时默认开 |

### 7.3 环境搭建踩坑

| 问题 | 原因 | 解决 |
|------|------|------|
| Python 3.10 不兼容 | LeRobot 0.6 要求 ≥3.12 | 用 Python 3.12 |
| 缺 accelerate / diffusers | `[pusht, aloha]` 不够 | 加 `[diffusion, training]` |
| 缺系统 ffmpeg | torchcodec 依赖 | `apt install ffmpeg` |
| HF 网络不通 | 国内墙 | `HF_ENDPOINT=https://hf-mirror.com` |
| Xet CAS 401 | 镜像站不支持 Xet 传输 | `HF_HUB_DISABLE_XET=1` |
| gym_pusht 注册失败 | AsyncVectorEnv forkserver 子进程 | `--eval.use_async_envs=false` |
| MuJoCo OpenGL 错误 | headless 实例无显示器 | `xvfb-run -a` 虚拟显示 |
| 旧 Hydra 格式不能用 | LeRobot 0.6 改 argparse | 对照 Makefile 重写命令 |

---

## 八、面聊 FAQ (扩展)

### Q1: 什么是 Diffusion Policy？跟 Stable Diffusion 有什么关系？

两者底层原理相同（DDPM）。区别：SD 用文本条件生成图片，DP 用观测条件生成动作。从技术上讲，SD 是 latent diffusion（在 VAE latent space 里做），DP 是在原始动作空间直接做。DP 用 1D U-Net，SD 用 2D U-Net。

### Q2: 为什么用扩散模型做机器人控制，不用普通网络？

直接回归 MSE Loss 训练的网络，遇到多解问题会输出"平均值"。扩散模型通过迭代去噪 + 随机初始噪声，可以覆盖多模态分布。

### Q3: ACT 和 Diffusion Policy 有什么区别？

ACT 是 VAE + Transformer，一次前向出动作；DP 是扩散模型，100 步迭代出动作。ACT 推理快但表达式能力不如扩散；DP 更精准但推理慢。两者代表了具身智能的两个技术路线：效率派 vs 质量派。

### Q4: 为什么你的成功率比论文低？

训练步数少（100K vs 200K+），图像增强关了，默认超参没调。这是有意选择——先跑通基线再优化。

### Q5: 你具体做了什么？

从零在云 GPU 上搭环境，下载数据集，用 LeRobot 跑 Diffusion Policy + ACT 训练，评估 5 个 checkpoint，画训练曲线，分析结果。全程独立完成，包括解决网络、依赖、OpenGL 等各种问题。

### Q6: 模型参数和训练时间？

DP: 263M 参数, 2.5h。ACT: 52M 参数, ~2h。都在单卡 RTX 4090。

### Q7: 下一步？

开启图像增强重训 DP → 目标 50-60%；ACT 跑完后做对比分析；阅读更多 VLA 相关工作 (RT-2, Octo, π0)。

---

## 九、关键代码速查

| 你想找... | 文件 | 位置 |
|-----------|------|------|
| 训练主循环 | `lerobot_train.py` | `train()` |
| DP forward + loss | `modeling_diffusion.py` | `compute_loss()` L334 |
| DP 动作推理 | `modeling_diffusion.py` | `conditional_sample()` L233 |
| 视觉编码器 | `modeling_diffusion.py` | `DiffusionRgbEncoder` L473 |
| 1D U-Net | `modeling_diffusion.py` | `DiffusionConditionalUnet1d` L627 |
| ACT forward + loss | `modeling_act.py` | `forward()` |
| 训练配置 | `train.py` | `TrainPipelineConfig` L78 |
| 评估入口 | `lerobot_eval.py` | — |
