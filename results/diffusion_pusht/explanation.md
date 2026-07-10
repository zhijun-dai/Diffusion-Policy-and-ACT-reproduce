# Diffusion Policy 复现：完整讲解

> 这份文档帮你理解：我们做了什么、代码怎么跑的、原理是什么、面聊时怎么回答。

---

## 一、背景：模仿学习与 Diffusion Policy

### 1.1 问题设定

机器人学里有个基本问题：**给定一张图（或传感器数据），让机器人执行正确的动作**。

传统做法是手写规则：看到 block 在左边 → 往右推。但这遇到新场景就废了。

**模仿学习 (Imitation Learning)** 换个思路：给机器人看一堆"专家示范"（人类遥控机械臂完成任务），让它自己学"看到什么画面 → 做什么动作"的映射。不需要手写规则。

这其实是监督学习——输入观测、输出动作，MSE Loss 训练。在 LeRobot 框架里就叫 **Behavior Cloning**。

### 1.2 Diffusion Policy 的改进

普通模仿学习有个问题：直接回归动作坐标，模型倾向于输出"平均动作"，做不到精细操作。

Diffusion Policy 的解法：
- 把**动作生成**建模为一个**去噪过程**（和 Stable Diffusion 生成图片的原理一样）
- 训练时：给真实动作加噪声 → 让模型学"预测噪声"
- 推理时：从纯噪声开始 → 逐步去噪 → 得到干净的动作序列

核心优势：能建模复杂的多模态动作分布（比如说"推左边也行、推右边也行"），不会坍缩成"平均动作"。

### 1.3 PushT 任务

PushT 是一个简化版 2D 仿真任务：
- 桌上有一个 **T 形方块**
- 目标：把方块推到 **指定的位置和角度**
- 输入：俯视 RGB 图像 (384×384) + 末端执行器位置 (2维)
- 输出：末端执行器位移 (2维，x 和 y)

这是 Diffusion Policy 论文的标准测试环境。简单、跑得快、适合验证。

---

## 二、训练流程全景

### 2.1 数据流

```
PushT 数据集 (206 episodes, 25650 frames)
    │
    ▼
DataLoader (batch_size=8)
    │  每个样本包含:
    │  - observation.state: (B, 2, 2)    [2步观测, 每步2维: agent x,y]
    │  - observation.image: (B, 2, 3, 384, 384)  [2步观测, RGB图像]
    │  - action: (B, 64, 2)            [64步未来动作]
    │  - action_is_pad: (B, 64)         [标记哪些步是padding]
    │
    ▼
DiffusionPolicy.forward()
    │
    ├── 1. ResNet-18 编码图像 → 32个关键点坐标 (SpatialSoftmax)
    ├── 2. 拼接 [图像特征, agent位置] 作为 condition
    ├── 3. 给 action 加噪声 (forward diffusion)
    ├── 4. U-Net 预测噪声 (denoising)
    └── 5. MSE Loss: 预测噪声 vs 真实噪声
```

### 2.2 训练循环 (关键代码对应 `lerobot_train.py`)

每轮训练做的事（简化版）：

```python
for batch in dataloader:
    # 1. 前向传播 + 计算 loss
    loss, _ = policy.forward(batch)   # → modeling_diffusion.py:162
    
    # 2. 反向传播
    accelerator.backward(loss)
    
    # 3. 梯度裁剪 (防止梯度爆炸)
    clip_grad_norm_(policy.parameters(), max_norm=10.0)
    
    # 4. 更新参数
    optimizer.step()
    lr_scheduler.step()
```

### 2.3 Loss 计算详解 (`modeling_diffusion.py:334`)

```python
def compute_loss(self, batch):
    # Step 1: 用 ResNet 编码图像 → 提取视觉特征
    global_cond = self._prepare_global_conditioning(batch)
    # global_cond shape: (B, 384)  [256维图像特征 + 4维agent位置] × 2步
    
    # Step 2: 给真实动作加噪声
    trajectory = batch["action"]       # (B, 64, 2) 真实动作序列
    eps = torch.randn(trajectory.shape)  # 随机噪声
    timesteps = torch.randint(0, 100, (B,))  # 随机选去噪步数
    
    # 加噪: 噪声越大, timestep越大
    noisy_trajectory = self.noise_scheduler.add_noise(
        trajectory, eps, timesteps
    )
    
    # Step 3: U-Net 预测噪声
    pred = self.unet(noisy_trajectory, timesteps, global_cond)
    # U-Net 输入: (加噪后的动作, 当前时刻, 观测条件)
    # U-Net 输出: (预测的噪声, 形状同 trajectory)
    
    # Step 4: 算 MSE
    loss = F.mse_loss(pred, eps)  # 预测噪声 vs 真实噪声
    return loss
```

**直觉理解**：U-Net 看到"加噪后的动作轨迹"，加上"画面和状态"，要猜出"噪声长什么样"。学会去噪 = 学会生成正确的动作。

---

## 三、模型架构详解

### 3.1 整体结构

```
DiffusionPolicy
└── DiffusionModel
    ├── rgb_encoder: ResNet-18 (去掉最后两层)
    │   ├── SpatialSoftmax: 找出特征图的"重心"
    │   └── Linear(64 → 64)
    │
    ├── unet: DiffusionConditionalUnet1d
    │   ├── down_modules: 编码器 (逐步降采样)
    │   ├── mid_modules: 瓶颈层
    │   ├── up_modules: 解码器 (逐步升采样 + skip connections)
    │   └── final_conv: 输出映射
    │
    └── noise_scheduler: DDPMScheduler (100步)
```

### 3.2 视觉编码器 (`DiffusionRgbEncoder`)

```python
# ResNet-18 backbone → 特征图 (512, 12, 12)
features = self.backbone(image)  

# SpatialSoftmax: 找每个通道的激活中心 → 32个关键点, 每个2D坐标
keypoints = self.pool(features)  # shape: (32, 2)

# Flatten + Linear
output = self.relu(self.out(keypoints.flatten()))  # → 64维
```

SpatialSoftmax 是什么？不同于普通池化取最大值，它在特征图上做 softmax 然后算加权重心。这相当于让网络自己学"关注图像哪里"——对于机器人来说，关注 T 形方块的位置比关注背景更有意义。

### 3.3 1D U-Net (`DiffusionConditionalUnet1d`)

输入/输出都是动作序列的形状 `(B, 64, 2)`——64步、每步2维。

核心是用 **1D 卷积** 处理时间维度的动作序列（不是 2D 卷积处理图像）。编码器-解码器结构 + skip connections。

**FiLM 条件注入**：观测条件通过 FiLM (Feature-wise Linear Modulation) 注入 U-Net 每一层:
```python
# cond: 拼接了 timestep_embedding + 视觉/状态特征
cond_embed = self.cond_encoder(cond)       # → (B, out_channels)
out = out + cond_embed                     # FiLM bias modulation
```

这等价于告诉网络："在这些观测下，去噪应该往哪个方向走"。

### 3.4 推理时怎么生成动作 (`generate_actions`)

```python
# 1. 编码观测 → condition
global_cond = self._prepare_global_conditioning(batch)

# 2. 从纯噪声开始
action = torch.randn(B, 64, 2)

# 3. 逐步去噪 (100步 → 0步)
for t in [99, 98, ..., 0]:
    pred_noise = self.unet(action, t, global_cond)
    action = scheduler.step(pred_noise, t, action)  # 去掉一点噪声

# 4. 取前 32 步作为实际执行的动作
return action[:, :32]
```

进一步优化：用 DDIM 调度器可以跳到比如只用 10 步去噪（牺牲一点点质量换速度）。

---

## 四、我们的实验配置

### 4.1 训练参数

| 参数 | 值 | 含义 |
|------|-----|------|
| batch_size | 8 | 每批 8 个样本 |
| steps | 100,000 | 总共训练步数 |
| n_obs_steps | 2 | 用前 2 帧观测 |
| horizon | 64 | 预测未来 64 步动作 |
| n_action_steps | 32 | 实际执行 32 步 |
| num_train_timesteps | 100 | 扩散 100 步 |
| optimizer | Adam | 优化器 |
| lr | 1e-4 → 0 (cosine) | 学习率衰减 |
| vision_backbone | resnet18 | 图像编码器 |
| down_dims | [512, 1024, 2048] | U-Net 各层通道数 |

### 4.2 数据细节

- PushT 数据集：206 个 episodes，25650 帧
- 每个 episode 约 125 帧 (~12.5秒)
- 有效 batch size = 8 × 1 GPU = 8

---

## 五、结果解读

### 5.1 Loss 曲线

| Step | Loss | 解读 |
|------|------|------|
| 200 | 0.390 | 刚开始，预测噪声还很差 |
| 20K | 0.027 | 已经学会基本的去噪 |
| 100K | 0.010 | 接近收敛 |

Loss 是预测噪声 vs 真实噪声的 MSE。0.01 意味着模型平均每个维度误差 ~0.1 个标准差——对去噪来说很好了。

### 5.2 成功率曲线

| Checkpoint | 成功率 | 解读 |
|------------|--------|------|
| 20K | 0% | 还不会推 |
| 40K | 6% | 偶尔能推到 |
| 60K | 24% | 成功率快速上升 |
| 80K | 36% | 接近收敛 |
| 100K | 36% | 平稳 |

成功率是 50 次评估中 T 形方块到达目标位置的百分比。36% 距离论文报告的 70-80% 有差距，原因：
- 训练步数少（100K vs 论文的 200K+）
- 图像增强关了（论文用 ColorJitter、RandomAffine 等防止过拟合）
- 用的是默认超参，没调参

### 5.3 VRAM 使用

- 训练时 ~5GB / 24GB——4090 绰绰有余
- 模型 263M 参数，中等规模

---

## 六、面聊 FAQ

### Q1: 什么是 Diffusion Policy？跟 Stable Diffusion 有什么关系？

**答**：Stable Diffusion 是用来"从文字生成图片"的扩散模型——从噪声逐步去噪得到图片。Diffusion Policy 把这个思路用到**动作生成**——从噪声逐步去噪得到机器人动作轨迹。核心区别是：SD 用文本作为条件（text → image），Diffusion Policy 用**图像观测和机器人状态**作为条件（observation → action）。

### Q2: 为什么用扩散模型做机器人控制，而不用简单的神经网络直接回归？

**答**：直接回归动作坐标有个问题——当存在多种合理动作时（比如可以从左边绕，也可以从右边绕），模型会输出"平均"动作，导致失败。扩散模型通过迭代去噪可以建模多模态分布，生成的动作更精准、更多样。

### Q3: 只用 2 帧历史观测够吗？

**答**：对 PushT 这种低速准静态任务够用。2 帧能捕捉物体的运动方向。更复杂的动态任务可能需要更多帧。论文里也用了 2 帧。

### Q4: 为什么成功率只有 36%，没有达到论文的 70-80%？

**答**：几个原因：1) 只训练了 100K 步，论文训 200K+；2) 关闭了图像数据增强（论文用它防止过拟合）；3) 没调超参。这些都是有意为之——先跑通基线，验证环境正确，后续可以轻松改进。

### Q5: 整个流程你具体做了什么？

**答**：1) 在云端租 RTX 4090 GPU 实例；2) 从零搭建 conda 环境，装 PyTorch + LeRobot；3) 下载 PushT 数据集，用默认配置跑 Diffusion Policy 训练；4) 每 2 万步保存 checkpoint，训练 10 万步；5) 对每个 checkpoint 做 50 次评估，统计成功率；6) 画 loss 曲线和成功率曲线，分析结果。

### Q6: 模型有多少参数？训练了多久？

**答**：Diffusion Policy 模型 2.63 亿参数（主要是 ResNet-18 视觉编码器约 11M + U-Net 约 252M）。在单卡 RTX 4090 上训练 10 万步耗时约 2.5 小时。VRAM 占用约 5GB。ACT 模型更小，约 5200 万参数。

### Q7: 下一步怎么改进？

**答**：三个方向：1) 开启图像增强重新训练，预期能提升到 50-60%；2) 延长训练到 200K+ 步；3) 尝试 ACT 模型做对比实验。同时也计划在 ALOHA 双臂任务上复现 ACT。

---

## 七、关键代码速查

| 你想找... | 文件 | 函数/行号 |
|-----------|------|-----------|
| 训练主循环 | `lerobot_train.py` | `train()` |
| Forward pass + loss | `modeling_diffusion.py` | `compute_loss()` L334 |
| 动作推理/去噪 | `modeling_diffusion.py` | `conditional_sample()` L233 |
| 视觉编码器 | `modeling_diffusion.py` | `DiffusionRgbEncoder` L473 |
| 1D U-Net | `modeling_diffusion.py` | `DiffusionConditionalUnet1d` L627 |
| 训练配置 | `train.py` | `TrainPipelineConfig` L78 |
