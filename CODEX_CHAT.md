# Codex 对话与讲解记录

> 建立日期：2026-07-22  
> 工作目录：`/root/autodl-tmp/diffusers`

## 沟通约定

从现在开始，本次对话中的详细讲解、公式、代码分析和重要结论都记录在这个文件中。

- 聊天对话框只保留简短提示和这个文件的链接。
- 新内容会继续写入或更新到本文件。
- 公式优先使用 Markdown/LaTeX 格式。
- 代码中的变量名、文件名和行号会尽量保留，方便对照源码。

---

# Flux2 Klein Img2Img 训练中的 loss 计算

分析文件：

`examples/dreambooth/train_dreambooth_lora_flux2_klein_img2img.py`

## 1. 最重要的结论

按照当前工作区里的实际代码，默认 loss 是：

> 模型猜出的“移动方向”，与正确的“移动方向”之间的平均平方误差。

默认情况下可以写成：

$$
\text{loss}
=
\operatorname{mean}\left[
\left(
\text{model\_pred}
-
(\text{noise}-\text{model\_input})
\right)^2
\right]
$$

模型在这里：

- 不是直接预测最终图片；
- 不是单纯预测随机噪声；
- 而是预测一根“从干净图片指向随机噪声的方向箭头”。

---

## 2. 把变量翻译成人话

| 代码变量 | 最通俗的含义 |
|---|---|
| `model_input` | 目标图片压缩后得到的“数字暗号” |
| `cond_model_input` | 条件图片的“数字暗号”，相当于参考图 |
| `noise` | 一堆随机雪花数字 |
| `sigmas` | 决定加入多少噪声的旋钮 |
| `noisy_model_input` | 目标图暗号与噪声混合后的结果 |
| `model_pred` | 模型猜出的移动方向 |
| `target` | 老师提供的正确移动方向 |
| `weighting` | 这道题的评分权重 |
| `loss` | 模型最终错了多少 |

特别容易混淆的一点：

> 数据集里的“目标图片”会变成 `model_input`；第 1737 行名叫 `target` 的变量已经不是图片，而是 loss 使用的“标准方向答案”。

---

## 3. 第一步：把图片翻译成数字暗号

对应主训练脚本第 1650～1665 行。

```python
model_input = vae.encode(pixel_values).latent_dist.mode()
cond_model_input = vae.encode(cond_pixel_values).latent_dist.mode()
```

这里有两张图片：

- `pixel_values`：模型最终应该生成的目标图片；
- `cond_pixel_values`：提供给模型参考的条件图片。

可以暂时把 VAE 理解成一个“图片翻译器”：

```text
普通 RGB 图片
    ↓ VAE
模型更容易处理的一大堆数字
```

这些数字通常叫作“潜变量”或 `latent`。

后面的 `_patchify_latents` 和标准化，可以简单理解成：

```text
重新整理数字的摆放方式，
并把数字大小调整到模型习惯的范围。
```

---

## 4. 第二步：制造随机噪声

对应第 1677 行：

```python
noise = torch.randn_like(model_input)
```

意思是制造一份与目标图暗号形状完全相同的随机数字。

为了简化后面的公式，我们把：

$$
\text{model\_input}=x
$$

$$
\text{noise}=\epsilon
$$

其中：

- $x$ 是干净目标图片的数字暗号；
- $\epsilon$ 是随机噪声。

---

## 5. 第三步：决定这次把图片弄脏多少

第 1680～1695 行会为每张图片随机选择一个时间点，并得到对应的 `sigma`。

这里的 `timestep` 不是“训练已经进行到第几步”，而是：

> 这道训练题中，目标图片被噪声污染到了什么程度。

可以把 `sigma` 看成一个“噪声旋钮”：

- $\sigma$ 接近 0：几乎是干净图片；
- $\sigma$ 接近 0.5：图片和噪声各占一部分；
- $\sigma$ 接近 1：几乎是纯随机噪声。

---

## 6. 第四步：混合目标图与噪声

对应第 1696 行：

```python
noisy_model_input = (1.0 - sigmas) * model_input + sigmas * noise
```

写成公式就是：

$$
z_{\sigma}=(1-\sigma)x+\sigma\epsilon
$$

翻译成人话：

```text
混合结果
= 干净目标图 × 剩余比例
+ 随机噪声 × 噪声比例
```

例如，假设：

```text
干净暗号 x = 2
随机噪声 ε = 8
sigma = 0.25
```

那么模型看到的混合结果是：

$$
z_{\sigma}
=(1-0.25)\times2+0.25\times8
=3.5
$$

---

## 7. 第五步：让模型做题

第 1698～1725 行会把下面这些信息一起送给 Transformer：

- 被噪声弄脏的目标图；
- 没有加噪的条件图；
- 文本提示词；
- 当前噪声程度。

条件图相当于模型做题时可以查看的“参考资料”。

第 1706～1707 行把“带噪目标图”和“条件图”连接起来：

```python
packed_noisy_model_input = torch.cat(
    [packed_noisy_model_input, packed_cond_model_input],
    dim=1,
)
```

模型输出以后，第 1727 行会裁掉条件图对应的输出：

```python
model_pred = model_pred[:, : orig_input_shape[1], :]
```

因此：

> loss 只检查目标图区域的预测，不直接检查条件图区域；但是条件图仍然会影响模型怎样预测目标图。

---

## 8. 第六步：制作 Flow Matching 的标准答案

对应第 1737 行：

```python
target = noise - model_input
```

也就是：

$$
\text{target}=\epsilon-x
$$

### 为什么正确答案是这个？

想象一条直线：

```text
A 点：干净目标图 x
B 点：随机噪声 ε
```

`sigma` 表示一辆小车从 A 点向 B 点走了多少。

小车的位置是：

$$
z_{\sigma}
=x+\sigma(\epsilon-x)
$$

其中：

$$
\epsilon-x
$$

就是“从 A 点指向 B 点的方向箭头”。

因此模型学习的问题是：

> 我现在位于干净图片与随机噪声之间的某个位置，这条直线接下来朝哪个方向延伸？

这根方向箭头就是这里所说的 flow 或 velocity。

主文件第 1716 行的注释写着：

```python
# Predict the noise residual
```

但这个注释并不准确。根据真正用于监督模型的 `target`，模型预测的是：

$$
\epsilon-x
$$

而不是单独的 $\epsilon$。

生成图片时虽然需要从噪声走回图片，但生成阶段会让 `sigma` 从大变小，相当于沿着这根箭头倒着走，因此两者并不矛盾。

---

## 9. 第七步：计算最终 loss

主训练脚本第 1740 行调用：

```python
loss = custom_loss(
    model_pred=model_pred,
    target=target,
    weighting=weighting,
)
```

真正的 loss 实现在：

`examples/dreambooth/custom_loss_function.py` 第 32～39 行。

```python
mse_loss = torch.mean(
    (
        weighting.float()
        * (model_pred.float() - target.float()) ** 2
    ).reshape(target.shape[0], -1),
    1,
)

loss = self.mse_weight * mse_loss.mean()
```

逐步翻译如下。

### 9.1 模型答案减去正确答案

```python
model_pred - target
```

### 9.2 把每个差值平方

```python
(model_pred - target) ** 2
```

平方有两个作用：

- 猜大了和猜小了都算错误；
- 错得越多，受到的惩罚增长得越快。

### 9.3 乘以这道题的重要程度

```python
weighting * 平方误差
```

### 9.4 对一张图片中的所有数字取平均

```python
.reshape(target.shape[0], -1)
然后沿着每张图片的全部数字求平均
```

### 9.5 对 batch 中的所有图片再次取平均

```python
mse_loss.mean()
```

### 9.6 乘以 `mse_weight`

当前创建方式是：

```python
CustomLoss(mse_weight=1.0, wave_weight=1.0)
```

由于 `mse_weight=1.0`，所以不会改变 loss 数值。

完整公式可以写成：

$$
L
=
\frac{1}{B}
\sum_{b=1}^{B}
\frac{1}{D}
\sum_{j=1}^{D}
w_b
\left[
p_{b,j}-(\epsilon_{b,j}-x_{b,j})
\right]^2
$$

其中：

- $B$：一个 batch 中的图片数量；
- $D$：一张图片压缩后包含的数字数量；
- $p$：模型预测的方向；
- $\epsilon-x$：正确方向；
- $w$：当前噪声时间点的权重。

代码中的 `.float()` 表示先把参与 loss 的数字转成 float32。当前训练使用 bf16 混合精度时，这样计算 loss 通常更加稳定。

---

## 10. 默认 weighting 是什么？

参数默认值是：

```python
--weighting_scheme="none"
```

因此默认情况下：

$$
w=1
$$

最终 loss 就退化为最普通的平均平方误差：

$$
L
=
\operatorname{mean}
\left[
p-(\epsilon-x)
\right]^2
$$

其他可选设置的通俗解释如下。

| 设置 | 时间点怎样抽取 | loss 怎样评分 |
|---|---|---|
| `none` | 均匀随机抽取 | 所有题权重为 1 |
| `logit_normal` | 某些噪声程度更容易被抽到 | 权重仍为 1 |
| `mode` | 用另一种分布改变出题频率 | 权重仍为 1 |
| `sigma_sqrt` | 均匀抽取 | 乘以 $\sigma^{-2}$ |
| `cosmap` | 均匀抽取 | 乘以 $\frac{2}{\pi(1-2\sigma+2\sigma^2)}$ |

可以把它们分成两类：

1. `logit_normal` 和 `mode`：让模型多做某些噪声程度的题；
2. `sigma_sqrt` 和 `cosmap`：出题仍然均匀，但某些题答错后扣分更多。

---

## 11. 一个数字的完整 loss 示例

假设：

```text
model_input = 2
noise = 8
sigma = 0.25
weighting = 1
```

模型收到的题目是：

$$
z_{\sigma}
=0.75\times2+0.25\times8
=3.5
$$

老师给出的正确方向是：

$$
\text{target}
=8-2
=6
$$

假设模型预测：

$$
\text{model\_pred}=5
$$

那么误差是：

$$
5-6=-1
$$

平方误差是：

$$
(-1)^2=1
$$

乘以权重：

$$
1\times1=1
$$

真实训练只是同时对成千上万个类似的数字进行计算，然后把结果全部取平均。

---

## 12. 当前代码中三个容易误会的地方

### 12.1 当前没有小波 loss

虽然主文件创建 `CustomLoss` 时写了：

```python
wave_weight=1.0
```

而 `custom_loss_function.py` 中也存在小波相关代码，但当前 `forward()` 完全没有调用它们。

因此当前实际 loss 只有加权 MSE，小波 loss 没有参与训练。

### 12.2 当前没有 mask loss

`weighting` 是每张图片对应的时间点权重，不是图片区域遮罩。

一张图片中的所有潜变量位置都会参与 loss。

### 12.3 当前没有 prior-preservation loss

当前 loss 没有额外的 class loss 或 prior loss 项。

DreamBooth 文件名并不代表当前训练代码一定启用了 prior preservation。

---

## 13. 反向传播从哪里开始？

第 1742 行：

```python
accelerator.backward(loss)
```

到这里才真正开始反向传播。

在此之前的全部过程，可以压缩成：

```text
把图片变成数字暗号
    ↓
随机选择噪声程度
    ↓
把目标图与随机噪声混合
    ↓
让模型预测移动方向
    ↓
制作正确的方向答案
    ↓
计算模型答案与正确答案之间的平均平方误差
    ↓
得到一个标量 loss
    ↓
开始反向传播
```

---

# 补充：除了“干净图、时间步、预测速度、正确速度”，还有哪些变量？

## 14. 结论

有。你注意到的四个量已经抓住了 Flow Matching 的主干，但要完整描述“出题、作答、评分”，还缺少四类关键变量：

1. **随机噪声** `noise`：路径的另一个端点；
2. **带噪目标潜变量** `noisy_model_input`：模型当前真正看到的题目；
3. **条件信息**：条件图、文本、位置 ID 和可选的 guidance；
4. **loss 权重** `weighting`：决定当前时间点的误差怎样计分。

最精简但完整的公式是：

$$
\boxed{
\begin{aligned}
z_\sigma &= (1-\sigma)x+\sigma\epsilon,\\
\hat v_\theta
&=f_\theta(z_\sigma,\sigma,c,q,\mathrm{IDs},g),\\
v^* &= \epsilon-x,\\
L &= \operatorname{mean}
\left[
w(\sigma)(\hat v_\theta-v^*)^2
\right].
\end{aligned}
}
$$

其中：

- $x$：干净目标图的标准化潜变量；
- $\epsilon$：随机噪声；
- $\sigma$：当前噪声比例；
- $z_\sigma$：带噪目标潜变量；
- $c$：干净条件图潜变量；
- $q$：文本条件；
- $\hat v_\theta$：预测速度；
- $v^*$：正确速度；
- $w(\sigma)$：loss 权重。

---

## 15. 实际上有两张“干净图像”

这是 Img2Img 训练中最容易遗漏的地方。

### 15.1 干净目标图

```python
pixel_values
model_input
```

记作 $x$。

它是模型应该生成的目标图。原始 RGB 图片经过 VAE、patchify 和标准化以后，才变成公式里的 $x$。

$x$ 不会作为一份独立答案直接交给 Transformer。它主要用来：

- 与噪声混合，制造模型输入 $z_\sigma$；
- 与噪声相减，制造正确速度 $\epsilon-x$。

### 15.2 干净条件图

```python
cond_pixel_values
cond_model_input
```

记作 $c$。

它是提供给模型参考的 Img2Img 条件图。它同样经过 VAE、patchify 和标准化，但当前代码没有给它加噪。

模型收到的图像 token 实际是：

$$
\operatorname{concat}(z_\sigma,c)
$$

所以更准确的说法是：

> 一张干净目标图 $x$ 用来制造训练题和标准答案；另一张干净条件图 $c$ 直接提供给模型作为参考。

---

## 16. 核心变量总表

| 符号 | 代码变量 | 作用 | 送入 Transformer？ | 直接参与 loss？ |
|---|---|---|---|---|
| $I_{\mathrm{tar}}$ | `pixel_values` | 原始 RGB 目标图 | 否 | 先变成 $x$ |
| $x$ | `model_input` | 标准化后的干净目标潜变量 | 通过 $z_\sigma$ 间接进入 | 用来构造 `target` |
| $I_{\mathrm{cond}}$ | `cond_pixel_values` | 原始 RGB 条件图 | 否 | 否 |
| $c$ | `cond_model_input` | 标准化后的干净条件潜变量 | 是 | 间接影响预测 |
| $\epsilon$ | `noise` | 高斯随机噪声 | 通过 $z_\sigma$ 间接进入 | 用来构造 `target` |
| $\sigma$ | `sigmas` | 干净目标与噪声的混合比例 | 通过 $z_\sigma$ 体现 | 可决定权重 |
| $t$ | `timesteps` | 模型的时间条件 | 是，传入 `timesteps / 1000` | 不直接做差 |
| $z_\sigma$ | `noisy_model_input` | 当前带噪目标潜变量 | 是 | 不作为标准答案 |
| $q$ | `prompt_embeds` | 文本语义条件 | 是 | 间接影响预测 |
| — | `text_ids` | 文本 token 的位置 ID | 是 | 间接影响预测 |
| — | `model_input_ids`、`cond_model_input_ids` | 图像 token 的空间/时序位置 ID | 是 | 间接影响预测 |
| $g$ | `guidance` | 可选的 guidance 条件 | 视模型配置而定 | 间接影响预测 |
| $\hat v_\theta$ | `model_pred` | 模型预测速度 | 模型输出 | 是 |
| $v^*$ | `target` | 正确速度 $\epsilon-x$ | 否 | 是 |
| $w(\sigma)$ | `weighting` | 当前时间点的评分权重 | 否 | 是 |
| $L$ | `loss` | 加权平方误差的平均值 | 否 | 最终标量 |

这里的“直接参与 loss”是指显式出现在最终 MSE 公式中。条件图和文本虽然不直接出现在 MSE 那一行，却会改变 $\hat v_\theta$，所以会间接改变 loss。

---

## 17. 最容易忽略的两个核心量

### 17.1 随机噪声 `noise`

```python
noise = torch.randn_like(model_input)
```

如果 $x$ 是路径的 A 点，那么 $\epsilon$ 就是 B 点。

没有 $\epsilon$，下面两样东西都无法确定：

$$
z_\sigma=(1-\sigma)x+\sigma\epsilon
$$

$$
v^*=\epsilon-x
$$

即使目标图和时间步完全相同，只要重新抽取一次 $\epsilon$，带噪输入和正确速度都会变化。

### 17.2 带噪输入 `noisy_model_input`

```python
noisy_model_input = (1.0 - sigmas) * model_input + sigmas * noise
```

模型并不是只看到“干净图和时间步”以后凭空猜速度。模型真正判断的是当前位置 $z_\sigma$。

因此，`noisy_model_input` 才是题目主体。模型结合它、时间条件、参考图和文本，输出预测速度。

---

## 18. `u`、`indices`、`timesteps` 和 `sigmas`

代码依次计算：

```python
u = compute_density_for_timestep_sampling(...)
indices = (u * num_train_timesteps).long()
timesteps = noise_scheduler_copy.timesteps[indices]
sigmas = get_sigmas(timesteps, ...)
```

这四个代码变量共同完成一件事：选择训练题位于路径的哪个位置。

| 变量 | 含义 |
|---|---|
| `u` | 首先抽取的随机数；分布由 `weighting_scheme` 等设置决定 |
| `indices` | scheduler 数组中的离散下标 |
| `timesteps` | 送给 Transformer 的时间标签 |
| `sigmas` | 用于混合目标与噪声的比例，也用于计算 loss 权重 |

在当前 `FlowMatchEulerDiscreteScheduler` 中，二者的调度关系是：

$$
t=\sigma N
$$

其中 $N$ 是 `num_train_timesteps`。

因此，`timesteps` 与 `sigmas` 表示同一个调度位置的两种数值形式，但用途不同：

- `timesteps` 用于告诉模型当前时间；
- `sigmas` 用于构造带噪输入与 loss 权重。

---

## 19. 标准化参数也是隐藏变量

代码先执行：

```python
model_input = (model_input - latents_bn_mean) / latents_bn_std
cond_model_input = (cond_model_input - latents_bn_mean) / latents_bn_std
```

因此公式中的干净目标潜变量实际是：

$$
x=
\frac{x_{\mathrm{VAE}}-\mu_{\mathrm{BN}}}
{s_{\mathrm{BN}}}
$$

条件图同理。

这意味着正确速度：

$$
v^*=\epsilon-x
$$

是在标准化后的潜空间坐标系中定义的，不是在 RGB 像素空间中定义的。

---

## 20. 训练前向与生成推理的区别

这里的 `transformer(...)` 更准确地叫一次训练前向计算。

训练时：

- 已知干净目标图 $x$；
- 随机抽取一个时间点；
- 构造这个时间点的一道题；
- 可以得到正确速度并计算 loss。

真正生成图片时：

- 不知道目标图 $x$；
- 因而没有 `target` 和 loss；
- 从噪声开始，在多个时间步反复调用 Transformer；
- scheduler 根据每一步的预测速度更新 latent；
- 最后由 VAE 解码得到图片。

所以生成阶段仍然有：

$$
z_t,\quad t,\quad c,\quad q,\quad \hat v_\theta
$$

但没有训练阶段才能得到的：

$$
x,\quad v^*=\epsilon-x,\quad L
$$

---

## 21. 最好记的五组角色

1. **路径两端**：干净目标 $x$、随机噪声 $\epsilon$；
2. **当前位置**：时间 $\sigma$、带噪 latent $z_\sigma$；
3. **作答条件**：条件图 $c$、文本 $q$、位置 IDs、guidance；
4. **两种速度**：预测速度 $\hat v_\theta$、正确速度 $v^*=\epsilon-x$；
5. **评分规则**：权重 $w(\sigma)$、最终 loss $L$。

这五组变量基本覆盖了当前训练脚本从制造模型输入到计算 loss 的完整数学流程。


---

# 利用九个变量设计新的 Flow Matching loss

## 22. 先纠正研究边界：TransNormal 不作为理论依据

你的担心是正确的。

`paper/` 目录里的 TransNormal 论文研究的是透明物体法线估计，它没有建立 Flow Matching 的训练目标、速度场、时间步加权或跨时间一致性理论。因此，下面关于新 loss 的判断**不引用 TransNormal 作为核心证据**。

后面的证据分成三层：

1. **Flow Matching / Rectified Flow 正式论文**：用于确认路径、速度目标和时间步加权；
2. **直接研究 Flow Matching 辅助损失的预印本**：可作为候选，但证据等级较弱；
3. **图像恢复或频率域论文**：只用于支持任务先验，不冒充 Flow Matching 理论。

---

## 23. 九个“工具”并不是九个互相独立的变量

把九个量写在一起：

$$
\boxed{
\begin{aligned}
v^* &= \epsilon-x,\\
z_\sigma &= (1-\sigma)x+\sigma\epsilon
          =x+\sigma v^*,\\
\hat v_\theta &=f_\theta(z_\sigma,\sigma,c,q),\\
w&=w(\sigma).
\end{aligned}}
$$

可以看到：

- 独立的数据或随机量主要是 $x,\epsilon,\sigma,c,q$；
- $v^*$、$z_\sigma$ 和 $w$ 都由前面的量派生；
- $\hat v_\theta$ 是唯一由模型产生、并把梯度传回 LoRA 的量。

令速度残差为：

$$
r_\theta=\hat v_\theta-v^*.
$$

当前基础 loss 就是：

$$
L_{\mathrm{FM}}
=
\mathbb E\left[w(\sigma)\|r_\theta\|_2^2\right].
$$

反向传播的本质是：

$$
\frac{\partial L}{\partial\theta}
=
\frac{\partial L}{\partial\hat v_\theta}
\frac{\partial\hat v_\theta}{\partial\theta}.
$$

$x,\epsilon,\sigma,z_\sigma,c,q,v^*,w$ 在这次反向传播中都被当作常量。它们的作用是改变模型面对的题目或改变“怎样评分”，但它们自己不接收梯度。

因此，设计新 loss 时应先问：

> 新项究竟增加了新的信息、新的误差几何、新的时间权重，还是只把同一个 MSE 换了一种写法？

---

## 24. 三类看似新的 loss，其实只是时间步重加权

### 24.1 预测干净图像的普通 MSE

由当前速度预测反推出干净 latent：

$$
\hat x=z_\sigma-\sigma\hat v_\theta.
$$

因为 $z_\sigma=x+\sigma v^*$，所以：

$$
\hat x-x
=
-\sigma(\hat v_\theta-v^*)
=-\sigma r_\theta.
$$

于是：

$$
\boxed{
\|\hat x-x\|_2^2
=
\sigma^2\|r_\theta\|_2^2.}
$$

如果加入：

$$
L=L_{\mathrm{FM}}+\lambda_x L_x,
$$

它等价于：

$$
L
=
\mathbb E\left[
w(\sigma)(1+\lambda_x\sigma^2)
\|r_\theta\|_2^2
\right].
$$

所以它不是独立监督项，而是更重视某些 $\sigma$ 的时间权重。

### 24.2 预测噪声的普通 MSE

反推出噪声：

$$
\hat\epsilon
=
z_\sigma+(1-\sigma)\hat v_\theta.
$$

严格有：

$$
\boxed{
\|\hat\epsilon-\epsilon\|_2^2
=
(1-\sigma)^2\|r_\theta\|_2^2.}
$$

它同样只是另一种时间步重加权。

### 24.3 预测路径上另一个位置的普通 MSE

从当前点预测任意另一时刻 $s$：

$$
\hat z_s
=
z_\sigma+(s-\sigma)\hat v_\theta.
$$

真实位置为：

$$
z_s=x+s v^*.
$$

因此：

$$
\boxed{
\|\hat z_s-z_s\|_2^2
=
(s-\sigma)^2\|r_\theta\|_2^2.}
$$

仍然没有新增约束。

甚至下面这个看似使用了条件图 $c$ 的变化量 loss：

$$
\| (\hat x-c)-(x-c)\|_2^2
$$

也会因为 $c$ 抵消而重新变成 $\sigma^2\|r_\theta\|_2^2$。

这组等价关系与 2026 年关于 Flow Matching 的 loss-space / weighting 分析一致；CVPR 2026 的 $x_0$-supervision 工作也明确把此类改动解释为有效时间步权重，而不是新标签。

---

## 25. 九个工具真正能提供的四种能力

| 能力 | 主要变量 | 能做什么 | 是否增加新监督信息？ |
|---|---|---|---|
| 改变训练测度 | $\sigma,w$ | 让某些噪声区间被抽到更多或得分更高 | 否 |
| 改变误差几何 | $\hat v,v^*,x,\epsilon$ | 分别约束方向、模长、频率或稳健误差 | 否，但增加归纳偏置 |
| 建立跨时间关系 | 两组 $\sigma,z_\sigma,\hat v$ | 约束同一路径不同时间的预测一致 | 增加关系信息 |
| 使用任务条件 | $c,q$ | 空间保留、条件对比或语义感知 | 通常需要 mask、负条件或外部模型 |

最重要的区别是：

- **新信息**：例如额外 mask、外部感知特征或额外标注；
- **新预测自由度**：例如独立端点预测头；它不增加外部标签，但能形成非退化的一致性约束；
- **新几何**：没有新标签，但让模型特别重视方向、高频或模长；
- **新关系**：把两个时间点或两个条件下的预测联系起来；
- **新权重**：只改变不同时间点或空间位置的训练强度。

这些都可能改善有限容量 LoRA 的优化，但证据强度与风险完全不同。

---

## 26. 候选一：局部速度方向 loss

### 26.1 定义

在每个 latent 空间位置 $i$，把通道维上的速度看成一个向量：

$$
L_{\mathrm{dir}}
=
\mathbb E_i\left[
w(\sigma)
\left(
1-
\frac{
\langle\hat v_i,v_i^*\rangle
}{
\|\hat v_i\|_2\|v_i^*\|_2+\delta
}
\right)
\right].
$$

总 loss：

$$
\boxed{
L=L_{\mathrm{FM}}+\lambda_{\mathrm{dir}}L_{\mathrm{dir}}.}
$$

### 26.2 它利用了什么

- $\hat v$：模型预测的方向；
- $v^*=\epsilon-x$：老师给出的正确方向；
- $w(\sigma)$：不同时间点的权重。

基础 MSE 同时混合“方向错误”和“长度错误”；余弦项单独强调方向，因此不能化成 MSE 的标量时间权重。

它没有增加新标签，但改变了 loss 的几何。理想共同最优点仍然是：

$$
\hat v=v^*.
$$

### 26.3 优点与风险

优点：

- 只需要当前这一次模型前向；
- 计算量和显存增量很小；
- 与“模型正在预测速度向量”这一语义直接一致；
- 很适合作为第一个独立消融项。

风险：

- 速度模长很小时，余弦梯度可能不稳定，应在 `float32` 中计算并设置 `eps`；
- 它约束的是 latent flow velocity 的方向，不是 RGB 图像或表面法线的角度；
- 权重过大可能让模型忽视模长，必须保留基础 MSE。

### 26.4 是否有直接论文先例

有，但目前最直接的证据仍很新。

2026 年的预印本 [ModaFlow: Modality-Aware Flow Matching for High-Fidelity Virtual Try-On](https://arxiv.org/abs/2606.27773) 在条件式 Flow Matching / LoRA 场景中加入了预测速度与目标速度的 cosine loss，并报告了对应消融。不过它是 2026 年 6 月的预印本，任务是虚拟试衣，不能把其结果直接等同于当前图像融合 LoRA。

因此，对这项候选的判断是：

> 数学上确实是新误差几何，工程成本低，也有直接 Flow Matching 先例；但跨任务收益仍需本地消融验证。

---

## 27. 候选二：跨时间端点一致性 loss

如果要求第二项也必须来自 Flow Matching 理论，那么最合适的不是小波，而是对同一条路径再取一个时间点。

### 27.1 构造同一路径上的两个题目

固定同一个：

$$
x,\epsilon,c,q,
$$

选择相邻的 $\sigma$ 和 $\sigma'$：

$$
z_\sigma=(1-\sigma)x+\sigma\epsilon,
$$

$$
z_{\sigma'}=(1-\sigma')x+\sigma'\epsilon.
$$

分别做两次预测：

$$
\hat v_\sigma
=f_\theta(z_\sigma,\sigma,c,q),
$$

$$
\hat v_{\sigma'}
=f_{\theta^-}(z_{\sigma'},\sigma',c,q).
$$

$\theta^-$ 可以是 stop-gradient 分支或 EMA teacher。

### 27.2 比较两个时间点预测出的干净端点

$$
\hat x_\sigma=z_\sigma-\sigma\hat v_\sigma,
$$

$$
\hat x_{\sigma'}=z_{\sigma'}-\sigma'\hat v_{\sigma'}.
$$

定义：

$$
\boxed{
L_{\mathrm{end}}
=
\left\|
\hat x_\sigma
-\operatorname{sg}(\hat x_{\sigma'})
\right\|_2^2.}
$$

也可以再加一个较弱的速度一致性项：

$$
L_{\mathrm{vel-cons}}
=
\left\|
\hat v_\sigma
-\operatorname{sg}(\hat v_{\sigma'})
\right\|_2^2.
$$

总目标可写成：

$$
L
=L_{\mathrm{FM}}
+\lambda_{\mathrm{end}}L_{\mathrm{end}}
+\lambda_{\mathrm{vel}}L_{\mathrm{vel-cons}}.
$$

第一轮实验建议只开 $L_{\mathrm{end}}$，不要同时引入两个未标定系数。

### 27.3 为什么这次不再等价于普通 MSE

单次预测的 $\hat x-x$ 会化成 $-\sigma r_\sigma$；但两个独立前向之间的端点差为：

$$
\hat x_\sigma-\hat x_{\sigma'}
=
-\sigma r_\sigma+\sigma' r_{\sigma'}.
$$

平方后含有两个时间点残差之间的交叉项，因此不能化成某个单独的 $w(\sigma)\|r_\sigma\|^2$。它真正建立了跨时间关系。

### 27.4 论文依据与边界

- [Consistency Models](https://proceedings.mlr.press/v202/song23a.html)，ICML 2023：建立了同一 ODE 轨迹上不同时间点应映射到同一端点的正式先例；它不是专门针对 FM 速度输出。
- [Consistency Flow Matching](https://arxiv.org/abs/2407.02398)，2024 arXiv 预印本：直接联合端点一致性与速度一致性，并使用目标网络；与这里最接近。
- [Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow](https://openreview.net/forum?id=XVjTT1nw5z)，ICLR 2023：Reflow 通过重新耦合与再训练让轨迹变直，但不是简单的 minibatch 辅助 loss。

关键警告：

- 单条条件插值路径的 $v^*=\epsilon-x$ 恒定，不代表许多路径叠加后的模型边际速度场必须处处恒定；
- 速度一致性只能是软正则，不能当作硬物理定律；
- 这类方法的主要证据常是降低轨迹弯曲或改善少步采样，不等于一定提升常规多步推理的图像细节；
- stop-gradient 第二分支仍需额外一次 Transformer 前向，训练耗时通常约增加到原来的 $1.3\sim1.5$ 倍；若两个分支都反传，则更接近 2 倍。

---

## 28. 候选三：高频速度残差 loss

这项更贴近当前图像编辑任务，但必须明确：

> 它是频率域工程先验，不是由 Flow Matching 理论必然推出的 loss。

令 $H$ 表示只保留 Haar 小波的 LH、HL、HH 三个高频子带：

$$
\boxed{
L_{\mathrm{HF}}
=
\mathbb E\left[
w(\sigma)
\|H(\hat v-v^*)\|_2^2
\right].}
$$

总目标：

$$
L=L_{\mathrm{FM}}+\lambda_{\mathrm{HF}}L_{\mathrm{HF}}.
$$

它的梯度为：

$$
\nabla_{\hat v}L_{\mathrm{HF}}
=2wH^\top H(\hat v-v^*),
$$

而基础 MSE 的梯度是：

$$
\nabla_{\hat v}L_{\mathrm{FM}}
=2w(\hat v-v^*).
$$

所以 $H^\top H$ 会专门放大空间快速变化的速度误差，属于新的频率归纳偏置。

必须注意：

- 若对完整正交 Haar 的低频与所有高频子带使用相同权重，根据 Parseval 定理，它会重新等价于原 MSE；
- 必须只取高频，或对子带使用不同权重；
- 当前张量是 patchify、标准化后的 latent 网格，其“高频”不完全等于 RGB 像素细节；
- 当前 `custom_loss_function.py` 已预留 `DWTForward` 与 `wave_weight`，但尚未真正把小波项加入最终 loss；
- Wavelet 相关图像论文只能支持“频率偏置可能有用”，不能证明它会改善当前 FM LoRA。

一个相关但非 FM 专用的正式工作是 [Training Generative Image Super-Resolution Models by Wavelet-Domain Losses Enables Better Control of Artifacts](https://openaccess.thecvf.com/content/CVPR2024/html/Korkmaz_Training_Generative_Image_Super-Resolution_Models_by_Wavelet-Domain_Losses_Enables_Better_CVPR_2024_paper.html)，CVPR 2024。它支持对子带不等权的图像生成训练思想，但不是当前速度残差公式的直接先例。

---

## 29. 候选四：速度模长 loss

还可以在每个位置单独比较速度模长：

$$
L_{\mathrm{mag}}
=
\mathbb E_i\left[
w(\sigma)g(\sigma)
\left(
\|\hat v_i\|_2-\|v_i^*\|_2
\right)^2
\right].
$$

其中 $g(\sigma)$ 可让它只在接近噪声端的区域较强。它与方向项互补：

- $L_{\mathrm{dir}}$ 管方向；
- $L_{\mathrm{mag}}$ 管长度；
- $L_{\mathrm{FM}}$ 保持完整逐元素锚点。

2026 年预印本 [The Velocity Deficit: Initial Energy Injection for Flow Matching](https://arxiv.org/abs/2605.14819) 认为条件平均会造成预测速度模长不足，并研究了 magnitude-aware Flow Matching。

但这项的风险高于方向 loss：它会有意偏离纯 MSE 的条件均值解，而且论文非常新。当前阶段更适合作为后续研究性消融，不建议和方向、高频项一次性同时加入。

---

## 30. 条件图 $c$ 与文本 $q$ 为什么不容易直接写进 loss

在当前单次正样本前向中，$c$ 和 $q$ 已经通过：

$$
\hat v=f_\theta(z_\sigma,\sigma,c,q)
$$

影响预测，但最终 MSE 无法判断模型到底“有没有认真使用条件”。

要显式利用它们，通常要增加关系或额外先验。

### 30.1 空间保留项

如果 $x$ 与 $c$ 严格空间对齐，可用它们的差异构造停止梯度的变化区域：

$$
m=\operatorname{sg}\left(
\mathcal N\bigl(\operatorname{mean}_{ch}|x-c|\bigr)
\right).
$$

在预计不应变化的区域，可约束：

$$
L_{\mathrm{pres}}
=
\|(1-m)\odot(\hat x-c)\|_1.
$$

它确实使用了条件图，但风险是 $x-c$ 也可能包含正常的光照或编码差异，错误 mask 会抑制必要编辑。若数据集中有真实编辑 mask，真实 mask 比从 $x-c$ 猜 mask 更可靠；那会成为第十个工具。

### 30.2 条件对比项

另一思路是比较正确条件 $(c,q)$ 与打乱条件 $(c^-,q^-)$ 下的预测，让正确条件更接近 $v^*$。

这需要：

- 第二次前向或足够大的 in-batch negatives；
- 确保负条件真的不匹配；
- 防止模型通过人为放大负样本误差来投机。

当前启动配置的 batch size 为 1，梯度累积也不会自动形成同一次前向中的负样本，因此不建议把它作为第一项实验。

---

## 31. 在发明新 loss 前，先检查当前时间步训练测度

当前代码中：

```python
--weighting_scheme="none"
```

是默认值，而且现有启动脚本没有覆盖它。因此当前实际是：

$$
\sigma\sim\operatorname{Uniform}(0,1),
\qquad
w(\sigma)=1.
$$

训练真正看到的时间重点是：

$$
\underbrace{p_{\mathrm{sample}}(\sigma)}_{\text{抽题频率}}
\times
\underbrace{w_{\mathrm{loss}}(\sigma)}_{\text{每题分值}}.
$$

二者必须一起分析。

在当前实现里：

- `logit_normal`、`mode`：改变 timestep 的抽样分布，loss 权重仍为 1；
- `sigma_sqrt`、`cosmap`：保持均匀抽样，改变 loss 权重；
- `none`：均匀抽样且权重为 1。

[Scaling Rectified Flow Transformers for High-Resolution Image Synthesis](https://arxiv.org/abs/2403.03206)，ICML 2024，系统研究了 Rectified Flow 的 timestep sampling，并发现偏向更有感知意义的中间噪声区间很重要。这也是当前 `logit_normal` 等选项的直接来源。

因此，即使你的最终目标是发明新损失，也应先做：

$$
\texttt{none}
\quad\text{vs.}\quad
\texttt{logit\_normal}
$$

的基线。否则辅助项带来的变化可能只是它间接改变了不同时间段的梯度比例。

Min-SNR（ICCV 2023）与 P2（CVPR 2022）也说明时间步权重会显著影响 diffusion 训练，但它们不是为当前线性 FM 路径直接设计的，只能作为需要重新推导和消融的参考：

- [Efficient Diffusion Training via Min-SNR Weighting Strategy](https://openaccess.thecvf.com/content/ICCV2023/html/Hang_Efficient_Diffusion_Training_via_Min-SNR_Weighting_Strategy_ICCV_2023_paper.html)
- [Perception Prioritized Training of Diffusion Models](https://openaccess.thecvf.com/content/CVPR2022/html/Choi_Perception_Prioritized_Training_of_Diffusion_Models_CVPR_2022_paper.html)

---

## 32. 最推荐的两条路线

### 路线 A：要求两项都尽量属于 Flow Matching

$$
\boxed{
L
=L_{\mathrm{FM}}
+\lambda_{\mathrm{dir}}L_{\mathrm{dir}}
+\lambda_{\mathrm{end}}L_{\mathrm{end}}.}
$$

- `direction`：单次前向、成本很低、直接改变速度误差几何；
- `endpoint consistency`：第二次前向、成本较高、真正增加跨时间关系。

这条路线理论叙事最干净，但端点一致性的收益更可能体现在轨迹平滑和少步推理，未必首先体现为局部画质提升。

### 路线 B：优先适配当前图像编辑 LoRA 的画质

$$
\boxed{
L
=L_{\mathrm{FM}}
+\lambda_{\mathrm{dir}}L_{\mathrm{dir}}
+\lambda_{\mathrm{HF}}L_{\mathrm{HF}}.}
$$

- `direction`：保留 Flow Matching 的直接速度几何依据；
- `HF residual`：把有限 LoRA 容量更多分配给边缘、接缝和局部快速变化。

这条路线改动小、算力低，也更贴合当前数据中“消除接缝、自然融合、保留人物与物品”的目标；但论文中必须诚实地把 HF 项称为任务驱动的频率正则，而不是 Flow Matching 定理。

---

## 33. 建议的实际消融顺序

不要一开始把两个新项一起打开，否则无法知道是谁产生了效果。

| 实验 | timestep 方案 | loss | 目的 |
|---|---|---|---|
| B0 | `none` | $L_{\mathrm{FM}}$ | 复现当前基线 |
| B1 | `logit_normal` | $L_{\mathrm{FM}}$ | 先测成熟的时间采样改动 |
| A1 | B0/B1 中较优者 | $L_{\mathrm{FM}}+\lambda_{dir}L_{dir}$ | 验证方向项 |
| A2 | 同一基线 | $L_{\mathrm{FM}}+\lambda_{HF}L_{HF}$ | 验证高频项 |
| A3 | 同一基线 | $L_{\mathrm{FM}}+\lambda_{end}L_{end}$ | 验证跨时间一致性 |
| A4 | 仅在单项有效后 | 两个有效项组合 | 检查互补性 |

每个实验至少同时记录：

- 基础 FM MSE；
- 辅助项的未加权原始数值；
- 每个 $\sigma$ 区间的 loss；
- 固定 seed、固定 prompt、固定推理步数下的验证图；
- 条件保持、目标编辑完成度、伪影与清晰度。

系数不应凭“看起来差不多”直接设为 1。更稳妥的方法是先观察若干步的原始量级，再让辅助项初始梯度或标量贡献只占总 loss 的较小比例，然后做至少三个数量级附近的 sweep。

---

## 34. 最终判断

1. 九个变量足以设计新的**误差几何**和**跨时间关系**，但不足以凭空产生新的外部监督信息。
2. $x$-MSE、$\epsilon$-MSE、单个其他路径点 MSE 都只是 velocity MSE 的时间步重加权，不应当作新方法。
3. 第一项最值得实验的是局部速度方向 loss：真正不等价、便宜、且已有直接 FM 预印本先例。
4. 第二项取决于研究目标：
   - 强调 Flow Matching 理论完整性：跨时间端点一致性；
   - 强调当前图像编辑画质与实现成本：高频速度残差。
5. 在任何自定义 loss 之前，都应先把当前 `none` 与 `logit_normal` 做成可靠基线。
6. 目前没有哪篇论文能保证这两项会让 Flux2 Klein Img2Img LoRA 变好；真正可信的结论必须来自严格单项消融。

### 核心参考文献

- Lipman et al., [Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747), ICLR 2023.
- Liu et al., [Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow](https://openreview.net/forum?id=XVjTT1nw5z), ICLR 2023.
- Esser et al., [Scaling Rectified Flow Transformers for High-Resolution Image Synthesis](https://arxiv.org/abs/2403.03206), ICML 2024.
- Song et al., [Consistency Models](https://proceedings.mlr.press/v202/song23a.html), ICML 2023.
- Yang et al., [Consistency Flow Matching](https://arxiv.org/abs/2407.02398), arXiv preprint, 2024.
- Sangare et al., [Improving Controllable Generation: Faster Training and Better Performance via $x_0$-Supervision](https://openaccess.thecvf.com/content/CVPR2026/html/Sangare_Improving_Controllable_Generation_Faster_Training_and_Better_Performance_via_x0-Supervision_CVPR_2026_paper.html), CVPR 2026.
- Gagneux et al., [Training Flow Matching: The Role of Weighting and Parameterization](https://arxiv.org/abs/2603.06454), ICLR 2026 DeLTa Workshop.
- Sai et al., [ModaFlow](https://arxiv.org/abs/2606.27773), arXiv preprint, 2026.
- Li et al., [The Velocity Deficit](https://arxiv.org/abs/2605.14819), arXiv preprint, 2026.

---

# 新 loss 能否改善 LPIPS、FID 与 ArcFace？

## 35. 直接结论

有希望，但不能期待同一个辅助 loss 自动、稳定地同时改善三项指标。

针对当前 `test_multi_gpu.py` 的首要判断是：

| 方案 | LPIPS $\downarrow$ | FID $\downarrow$ | ArcFace $\uparrow$ | 当前证据 |
|---|---|---|---|---|
| `logit_normal` | 间接、未知 | **较有希望** | 间接、未知 | 正式 Rectified Flow 论文 |
| 速度方向 cosine | **有希望** | **有希望** | 可能间接改善，但证据弱 | 最相近的是 2026 FM+LoRA 预印本 |
| 高频速度残差 | **较有希望** | 有一定希望，也可能因伪影变差 | 大概率变化小或不稳定 | 频率域正式论文是近邻证据，不是同一 loss |
| 跨时间端点一致性 | 50 步下可能很小 | 少步推理更有希望 | 基本未知 | 主要证据来自少步 FID |

如果只讨论高频速度残差：

$$
\boxed{
\text{预期改善可能性：LPIPS}>\text{FID}>\text{ArcFace}.}
$$

这不是说 ArcFace 一定不变，而是高频项没有显式告诉模型“必须保留同一个人的身份”。

---

## 36. 三项指标在当前脚本里究竟测了什么

### 36.1 LPIPS

`test_multi_gpu.py` 第 123～130 行计算：

$$
\operatorname{LPIPS}
=
\frac1N\sum_{i=1}^N
d_{\mathrm{VGG}}
\left(
I_i^{\mathrm{gen}},
I_i^{\mathrm{target}}
\right).
$$

特点：

- 生成图与对应目标图逐对比较；
- 使用整张图，而不是只比较接缝、人物或 face crop；
- 越低越好；
- 对空间位置、形状、纹理和感知特征都敏感。

所以，如果高频 loss 恢复的是**目标图中真实存在且位置正确**的边缘、阴影和纹理，LPIPS 有机会降低；如果产生过锐、ringing、checkerboard 或错误纹理，LPIPS 反而会升高。

### 36.2 FID

第 153～169 行计算两组整图 Inception 特征分布之间的距离：

$$
\operatorname{FID}
=
d_{\mathrm{Fr\acute echet}}
\left(
\{\phi(I_i^{\mathrm{gen}})\},
\{\phi(I_i^{\mathrm{target}})\}
\right).
$$

特点：

- 它不利用一一配对关系；
- 只比较两组图像的均值和协方差；
- 越低越好；
- 真实细节与自然纹理可能改善 FID，系统性高频伪影也可能明显恶化 FID。

### 36.3 ArcFace

第 133～136 行对生成图和目标图检测到的第一张脸计算：

$$
S_{\mathrm{Arc}}
=
\frac1N\sum_{i=1}^N
\cos
\left(
F(P(I_i^{\mathrm{gen}})),
F(P(I_i^{\mathrm{target}}))
\right),
$$

其中 $P$ 是人脸检测与对齐，$F$ 是 ArcFace embedding。

特点：

- 越高越好；
- 关注身份特征，不等于全图清晰度；
- 对背景接缝、物品保持和大部分衣服纹理并不敏感；
- 高频更清晰有时能让人脸更容易检测，但不代表身份 embedding 一定更接近。

[ArcFace](https://openaccess.thecvf.com/content_CVPR_2019/html/Deng_ArcFace_Additive_Angular_Margin_Loss_for_Deep_Face_Recognition_CVPR_2019_paper.html) 本身就是为了学习身份判别的角度特征，而不是作为通用图像清晰度指标。

### 36.4 三项指标与训练 loss 之间隔着一条长链

高频速度 loss 优化的是：

$$
H(\hat v-v^*).
$$

而测试指标看到的是：

$$
\hat v
\longrightarrow
\text{50 步 ODE 积分}
\longrightarrow
\text{VAE 解码}
\longrightarrow
\text{RGB 图像}
\longrightarrow
\text{LPIPS/FID/ArcFace}.
$$

因此不存在“高频 velocity residual 降低，就必然使某个 RGB 指标单调改善”的定理。只能建立机制假设，再用严格消融检验。

---

## 37. 高频速度残差对三项指标的具体预期

定义：

$$
L_{\mathrm{HF}}
=
\mathbb E\left[
w(\sigma)\|H(r_\sigma)\|_2^2
\right],
\qquad
r_\sigma=\hat v_\sigma-v^*.
$$

若把 $H$ 近似看成高频投影，则：

$$
L_{\mathrm{FM}}+\lambda_{\mathrm{HF}}L_{\mathrm{HF}}
\approx
\|r_{\mathrm{low}}\|_2^2
+(1+\lambda_{\mathrm{HF}})
\|r_{\mathrm{high}}\|_2^2.
$$

也就是说，它把有限的 rank-16 LoRA 容量更多分配给高频速度误差。

### 对 LPIPS

这是最有希望的一项，因为当前任务明确包含：

- 消除人物与背景之间的视觉接缝；
- 保持人物、衣服和接触物品细节；
- 重塑局部光影边界。

这些错误的一部分确实位于中高空间频率。

### 对 FID

方向不如 LPIPS 确定：

- 若生成图从模糊变得更自然、更接近真实纹理分布，FID 可能下降；
- 若模型开始拟合噪点、振铃或不真实锐化，FID 可能上升。

### 对 ArcFace

通常只能间接影响：

- 正面可能：脸部边缘更稳定、人脸检测成功率提高；
- 负面可能：模型把容量消耗在全图纹理，或错误重绘面部高频；
- 中性可能：ArcFace 主要依赖身份结构，微小纹理变化几乎不改变 embedding。

### 当前数据规模下的特殊风险

$v^*=\epsilon-x$ 中包含白噪声 $\epsilon$。在只有 80 张训练图、global batch size 1、200 updates 的条件下，强行放大所有位置、所有时间的高频残差，也会放大高频随机梯度。

因此可能出现：

$$
\text{训练 HF residual 下降}
\quad\text{但}\quad
\text{验证 HF residual 不下降},
$$

也就是只记住了随机细节。

### 最相近的实验证据

2024 年预印本 [FitDiT](https://arxiv.org/abs/2411.10499) 在人物虚拟试衣中对预测干净图的 RGB 频谱加入 frequency loss。其消融中：

| 设置 | LPIPS $\downarrow$ | FID $\downarrow$ |
|---|---:|---:|
| w/o frequency loss | 0.1239 | 22.6325 |
| full | **0.1130** | **20.7543** |

这说明频率监督在相近人物编辑任务中确实可能同时改善 LPIPS 和 FID。

但它与当前方案有两个重要差别：

1. FitDiT 在预测干净图的 RGB / DFT 空间计算；
2. 当前候选是在标准化 latent velocity 的 Haar 高频上计算。

中间还隔着 VAE 解码与多步积分，所以这只能算“支持研究方向”，不能视为对当前公式的直接验证。

正式发表的 [WHFL, WACV 2023](https://openaccess.thecvf.com/content/WACV2023/papers/Kim_WHFL_Wavelet-Domain_High_Frequency_Loss_for_Sketch-to-Image_Translation_WACV_2023_paper.pdf) 也显示额外 wavelet 高频约束可降低多种图像翻译模型的 FID，但同样不是 Flow Matching velocity loss。

---

## 38. 速度方向 loss 的指标证据反而更直接

在正确速度附近，写成：

$$
v^*=m u,
\qquad
r=r_{\parallel}u+r_{\perp}.
$$

则局部近似有：

$$
1-\cos(\hat v,v^*)
\approx
\frac{\|r_{\perp}\|_2^2}{2m^2}.
$$

所以 direction loss 主要放大“偏离正确方向的横向误差”，而不是速度模长误差。

最相近的工作是 2026 年预印本 [ModaFlow](https://arxiv.org/html/2606.27773)：

- 使用 FLUX.1-Fill；
- 使用 LoRA 微调；
- 任务是条件式人物图像生成；
- 显式加入 predicted velocity 与 target velocity 的 cosine loss。

其保持其他组件不变的消融结果为：

| 设置 | LPIPS $\downarrow$ | paired FID $\downarrow$ | unpaired FID $\downarrow$ |
|---|---:|---:|---:|
| w/o cosine | 0.063 | 6.115 | 9.535 |
| full | **0.053** | **5.282** | **8.041** |

因此，在目前找到的论文中，velocity direction 对 LPIPS/FID 的证据比“latent Haar velocity loss”更直接。

但仍有三项限制：

1. ModaFlow 是 2026 年 6 月预印本，尚不是成熟共识；
2. 它使用上万张训练图、rank-256 LoRA 和额外 perceptual flow regularizer，与当前 80 张、rank-16 不同；
3. 它没有报告 ArcFace 或其他 identity metric。

所以 direction loss 对 ArcFace 最多是通过姿态、几何和人物结构更稳定而间接改善，不能声称已有直接证据。

---

## 39. `logit_normal` 与跨时间一致性的指标预期

### 39.1 `logit_normal`

[Scaling Rectified Flow Transformers for High-Resolution Image Synthesis](https://arxiv.org/html/2403.03206)，ICML 2024，直接比较了 uniform Rectified Flow 与 logit-normal timestep sampling。

论文中的代表性结果包括：

| 数据 | uniform RF FID | logit-normal FID |
|---|---:|---:|
| ImageNet | 49.70 | **45.78** |
| CC12M | 94.90 | **89.91** |

它是四种候选中对 FID 最成熟、成本最低的依据。

但当前任务只有 200 次更新。默认参数下 logit-normal 会显著集中于中段、减少时间轴两端的样本：

$$
P(u<0.1)\approx1.4\%,
\qquad
P(0.4<u<0.6)\approx31.5\%.
$$

约 200 次更新中，单侧最外 10% 区间期望只抽到约 3 次，而 uniform 约 20 次。因此它可能：

- 更集中学习中段的语义融合与光影协调；
- 也可能削弱端点附近的结构初始化或末端细节修正；
- 对 LPIPS 和 ArcFace 的净效果需要本地验证。

### 39.2 跨时间端点一致性

这类方法最主要的理论收益是减少轨迹不一致和低 NFE 积分误差。

当前测试固定：

```python
NUM_INFERENCE_STEPS = 50
```

因此，它在当前三项指标上的收益可能比 4、8 或 16 步测试小得多。

一个可证伪预测是：

$$
\Delta_{4/8\ \mathrm{steps}}
>
\Delta_{16\ \mathrm{steps}}
>
\Delta_{50\ \mathrm{steps}}.
$$

如果 endpoint consistency 只在 50 步偶然改善，而 4/8 步没有更明显优势，就不能把收益解释成“轨迹一致性更好”。

[Consistency Flow Matching](https://arxiv.org/html/2407.02398) 提供了 FID 方面的相关证据，但主要是低分辨率、少步生成，没有 paired LPIPS 或 ArcFace 证据。因此它不应是当前第一轮指标优化的优先项。

---

## 40. 为什么 ArcFace 需要另一类监督

在前面四种方案中，direction 是最有可能间接帮助人物几何与身份的一项，但四者都没有直接最小化：

$$
1-\cos(F(I^{\mathrm{gen}}),F(I^{\mathrm{target}})).
$$

如果 ArcFace 是硬目标，存在两种更对齐的路线。

### 40.1 便宜的 face-region 速度重加权

预先为每张训练图生成固定的人脸区域 mask $M_{\mathrm{face}}$，映射到 latent 网格后定义：

$$
L_{\mathrm{face}}
=
\mathbb E\left[
w(\sigma)
\|M_{\mathrm{face}}\odot(\hat v-v^*)\|_2^2
\right].
$$

它仍然只是空间重加权，没有增加新标签，但明确把有限 LoRA 容量更多分配给脸部。相比全图高频，它更可能影响 ArcFace，且不需要训练时 VAE 解码或运行人脸网络。

人脸 mask 会成为现有九个变量之外的第十个工具。

### 40.2 直接的 ArcFace identity loss

从当前速度预测干净 latent：

$$
\hat x=z_\sigma-\sigma\hat v_\theta,
$$

解码并做预先确定的人脸对齐 $P$：

$$
\boxed{
L_{\mathrm{ID}}
=
1-\cos
\left(
F(P(D(\hat x))),
F(P(I_{\mathrm{target}}))
\right).}
$$

其中：

- $D$：VAE decoder；
- $P$：固定、预先计算的 crop / alignment；
- $F$：冻结且可微的 PyTorch ArcFace backbone。

这与测试指标最直接对齐，但代价和风险明显更高：

- 每步需要 VAE decode 与 face backbone；
- 当前 evaluator 使用 InsightFace ONNX + NumPy，不能直接用于反向传播；
- 在线人脸检测/选框是不可微且可能漏检，训练前应预计算 crop；
- 高噪声时间点的 $\hat x$ 不可靠，identity loss 可能需要按 $\sigma$ 门控；
- 权重过大会造成脸部僵硬、过度复制或编辑不完整。

正式论文 [PREIM3D, CVPR 2023](https://openaccess.thecvf.com/content/CVPR2023/papers/Li_PREIM3D_3D_Consistent_Precise_Image_Attribute_Editing_From_a_Single_CVPR_2023_paper.pdf) 使用冻结 ArcFace 特征定义 identity cosine loss，并通过消融提高 identity consistency。这说明想稳定提高 ArcFace，显式 identity supervision 比全图高频更符合目标。

[Diffuse and Restore, WACV 2024](https://openaccess.thecvf.com/content/WACV2024/html/Suin_Diffuse_and_Restore_A_Region-Adaptive_Diffusion_Model_for_Identity-Preserving_Blind_WACV_2024_paper.html) 也用身份区域 guidance 改善身份保持。

因此：

> 如果限制在原来的九个变量、单次前向和低额外算力内，可以期待 ArcFace 间接变化，但不能设计出与 ArcFace 真正对齐的监督。

---

## 41. 当前评估脚本比 loss 选择更先构成瓶颈

### 41.1 FID 只有 10 张测试图

当前：

$$
N_{\mathrm{test}}=10,
\qquad
d_{\mathrm{Inception}}=2048.
$$

每组样本协方差的秩最多只有：

$$
\operatorname{rank}(\hat\Sigma)\le N-1=9.
$$

用 10 个样本估计 2048 维分布的均值和协方差非常不稳定。当前 FID 可以作为调试数字，但不足以判断两个接近的 loss 谁更好。

[Effectively Unbiased FID and Inception Score and Where to Find Them](https://openaccess.thecvf.com/content_CVPR_2020/html/Chong_Effectively_Unbiased_FID_and_Inception_Score_and_Where_to_Find_CVPR_2020_paper.html)，CVPR 2020，证明有限样本 FID 存在模型相关偏差；即使不同模型使用相同样本数，也不能消除这种偏差。

### 41.2 推理没有固定随机种子

当前 pipeline 调用没有传：

```python
generator=...
```

全脚本也没有为每个样本固定生成 seed。

因此不同 LoRA、不同运行次数或不同 GPU 数量得到的初始噪声可能不同。对只有 10 张图的测试集，这种随机差异足以掩盖小幅 loss 收益。

### 41.3 ArcFace 可能漏检或匹配错误人物

当前代码直接执行：

```python
face_app.get(image_cv2)[0]
```

这意味着：

- 没检测到脸时直接 `IndexError`；
- 多张脸时取检测器返回的第一张，未保证是主体人物；
- 没有记录 face detection rate。

历史评估中 checkpoint-400 和多个后续 checkpoint 正是因为生成图检测不到脸而失败。也就是说，严重失败的模型不会得到一个很低的 ArcFace，而是整次评估直接没有分数。

### 41.4 mask 被读取但完全没有参与

第 106 行读取了：

```python
mask_image = Image.open(mask_image_path)
```

但它：

- 没有传给 pipeline；
- 没有用于 LPIPS；
- 没有用于 FID；
- 没有用于 ArcFace。

所以当前三项指标无法区分：

- 接缝区域是否真正改善；
- 应保留区域是否被改坏；
- 全图背景相似度是否掩盖了局部失败。

### 41.5 一个样本还发生了轻微尺寸重采样

10 个样本中有一张输入/目标是 $834\times1248$，pipeline 输出为 $832\times1248$。脚本随后把目标图 resize 到输出尺寸，这会给该样本的 LPIPS 和特征指标引入额外重采样差异。

---

## 42. 历史 checkpoint 已提示过拟合可能比新 loss 更重要

现有 `run_checkpoint_eval.log` 记录：

| checkpoint | LPIPS $\downarrow$ | FID $\downarrow$ | ArcFace $\uparrow$ |
|---:|---:|---:|---:|
| 100 | **0.4033** | **112.5247** | **0.4783** |
| 200 | 0.5388 | 158.4236 | 0.4254 |
| 300 | 0.5957 | 223.5658 | 0.3662 |

从 100 到 200：

- LPIPS 约恶化 33.6%；
- FID 约恶化 40.8%；
- ArcFace 约下降 11.1%。

三项同时恶化，结合只有 80 张训练图、batch size 1、学习率 $10^{-4}$，强烈提示继续训练可能在过拟合或破坏预训练先验。

但由于这些 checkpoint 当时使用的推理 seed 没有固定，不能把这张表视作严格因果证据。正确做法是先用相同的逐样本初始噪声重测 checkpoint-100 与 checkpoint-200。

如果固定 seed 后 checkpoint-100 仍稳定胜出，那么：

> 早停、学习率和 LoRA 容量可能比新增 loss 带来更大的指标收益。

辅助 loss 仍可能作为正则延缓过拟合，但不能替代 checkpoint 选择。

---

## 43. 先建立一个能判断 loss 是否有效的评估协议

### 43.1 固定生成随机性

对每个 stem 使用固定且跨进程稳定的 seed，使所有模型在同一个样本上从完全相同的初始噪声开始。

最好同时采用多个固定生成 seed，例如：

$$
\{s_1,s_2,s_3,s_4,s_5\},
$$

然后报告均值与方差，而不是只跑一次。

### 43.2 对 paired 指标做 paired 统计

LPIPS 和 ArcFace 都有一一对应的目标图。对模型 A、B 应计算每张图、每个生成 seed 的差值：

$$
\Delta_i
=
M_i^{(B)}-M_i^{(A)},
$$

再报告 paired bootstrap confidence interval，而不只比较两个平均数的小数点后四位。

### 43.3 暂时不要让 10-image FID 决定胜负

在测试集扩展到至少数百张以前：

- FID 只作描述性参考；
- 不根据很小的 FID 差异宣布新 loss 有效；
- 优先看固定 seed 的 paired LPIPS、ArcFace、face detection rate 和人工检查。

### 43.4 增加不替代主指标的诊断项

可以保留原三项排行榜，同时增加：

- face detection rate；
- 编辑区域 LPIPS；
- 保留区域 LPIPS；
- 接缝边界带 LPIPS / gradient error；
- 每个样本而不是只有平均值；
- 4/8/16/50 步分别评估 endpoint consistency。

这些诊断项能告诉我们“为什么指标变化”，不会把全图平均数误解释成接缝质量。

---

## 44. 面向三项指标的最小可信实验顺序

### 第零阶段：先修正比较条件

1. 为每个测试样本固定相同生成 seed；
2. 重测 checkpoint-100 与 checkpoint-200；
3. 记录每张图的 LPIPS、ArcFace 与 face detection；
4. FID 暂时只作参考；
5. 若资源允许，至少使用 3 个训练 seed。

### 第一阶段：不增加复杂 loss

| 实验 | 目的 |
|---|---|
| U | `uniform/none + FM` 基线 |
| LN | `logit_normal + FM` |
| U-100 / U-200 | 判断早停是否比 loss 更重要 |

### 第二阶段：分别验证两个低成本候选

| 实验 | 主要假设 | 失败指纹 |
|---|---|---|
| U+HF | LPIPS 优先下降，FID 可能下降，ArcFace 不应显著退化 | train HF 降但 held-out 不降；过锐或 face 变形 |
| U+DIR | LPIPS/FID 下降，ArcFace 可能小幅上升 | 角度误差降但编辑不完整、模长误差升 |

不要第一轮就组合 HF 和 direction。

如果 HF 单项有效，再做：

$$
\{\mathrm{uniform},\mathrm{logit\ normal}\}
\times
\{\mathrm{HF\ off},\mathrm{HF\ on}\}
$$

的 $2\times2$ 消融，检查 timestep 分布与高频正则是否互相影响。

### 第三阶段：针对 ArcFace 决策

若 LPIPS/FID 改善而 ArcFace 不升：

1. 不要简单继续增大 $\lambda_{\mathrm{HF}}$；
2. 先加入便宜的 face-region residual；
3. 只有在 ArcFace 是核心硬目标且算力允许时，再考虑 decoded ArcFace identity loss。

### 第四阶段：最后测试 endpoint consistency

只在需要降低推理步数，或 4/8/16-step 指标也是目标时测试。若最终始终用 50 步，它的优先级低于 HF、direction、时间采样和早停。

---

## 45. 如何判断“真的同时改善了三项指标”

不要把三项不同量纲的数字随意相加成一个总分。更稳妥的是使用 Pareto / 非劣判定：

- LPIPS 的 paired mean 和置信区间下降；
- ArcFace 的 paired mean 上升，且 face detection rate 不下降；
- FID 在扩大测试集后下降；当前 10-image FID 只作支持性证据；
- 人工检查确认不是通过复制输入、减少必要编辑或产生锐化伪影换来的。

尤其要防止这种表面成功：

$$
\text{ArcFace 上升}
\quad\text{但}\quad
\text{人物没有完成必要的光影融合}.
$$

这可能只是模型更倾向复制条件图，而不是编辑能力真的增强。

---

## 46. 针对当前项目的最终建议

1. **高频速度残差值得做。**它最有希望先改善 paired LPIPS，对 FID 有合理但不确定的希望，对 ArcFace 只能要求“不退化”。
2. **direction loss 也应单独做。**目前同类 FLUX+LoRA 任务对 LPIPS/FID 的直接消融证据反而比 latent Haar 更强。
3. **不要先做 endpoint consistency。**当前用 50 inference steps，它的主要理论优势未必会体现在现有指标上。
4. **先固定测试 seed 并重测 checkpoint-100/200。**现有数据强烈提示早停可能是最大收益来源。
5. **若目标明确要求 ArcFace 上升，应增加 face-aware 监督。**全图高频并不是 identity loss。

最合理的近期路线是：

$$
\boxed{
\text{修正评估}
\rightarrow
\text{早停 / logit-normal 基线}
\rightarrow
\text{HF 与 direction 单项消融}
\rightarrow
\text{必要时加入 face-aware loss}.}
$$

---

## 47. 新问题的直接结论：SSIM、PSNR、KID 都可以加入

可以，而且这三项指标分别补充了不同信息。不过，它们不应和 LPIPS、ArcFace、FID
被当成六个地位相同的优化目标。

建议的分工是：

| 指标 | 方向 | 层级 | 当前 10 张测试图下的角色 |
|---|---:|---|---|
| LPIPS | $\downarrow$ | 配对、感知特征 | **主指标** |
| ArcFace cosine | $\uparrow$ | 配对、人脸身份特征 | **主指标**，同时报告人脸检测成功率 |
| SSIM | $\uparrow$ | 配对、局部结构 | **强辅助指标 / guard** |
| PSNR | $\uparrow$ | 配对、逐像素误差 | **诊断指标 / 弱 guard** |
| FID | $\downarrow$ | 集合、Inception 分布 | **辅助指标** |
| KID | $\downarrow$ | 集合、Inception 分布 | **辅助指标** |

这个定位和 ModaFlow 的评估思路基本一致：ModaFlow 在 paired evaluation 中报告
SSIM、LPIPS、FID 与 KID。PSNR 是我们额外加入的像素级诊断维度。

需要强调：

> “能正确计算”不等于“适合用来自动调参”。

在当前任务里，LPIPS 与 ArcFace 最接近最终目标；SSIM、PSNR、FID、KID 更适合告诉
我们模型是以什么方式变好或变坏。

---

## 48. SSIM 能补充什么

SSIM 比较局部窗口中的亮度、对比度和结构。其典型形式为：

$$
\operatorname{SSIM}(x,y)
=
\frac{(2\mu_x\mu_y+C_1)(2\sigma_{xy}+C_2)}
{(\mu_x^2+\mu_y^2+C_1)(\sigma_x^2+\sigma_y^2+C_2)}.
$$

其中：

- $\mu_x,\mu_y$ 表示局部均值；
- $\sigma_x^2,\sigma_y^2$ 表示局部方差；
- $\sigma_{xy}$ 表示局部协方差；
- $C_1,C_2$ 是避免分母不稳定的常数。

它与 LPIPS 的互补关系是：

- LPIPS 更偏向深层感知特征是否接近；
- SSIM 更直接检查轮廓、局部结构、位置和对比度是否保持。

因此，高频速度残差 loss 若真的改善边缘、纹理和局部结构，理论上可能同时使
LPIPS 降低、SSIM 升高。ModaFlow 的 direction loss 若改善几何一致性，也可能使
SSIM 上升。

但 SSIM 有两个局限：

1. 它对一两个像素的平移、resize 和配准误差很敏感；
2. 它有时会偏爱较平滑的图像，不能单独代表“视觉上更真实”。

所以 SSIM 适合作为 LPIPS 的结构补充，不适合替代 LPIPS。

---

## 49. PSNR 能补充什么

对于取值范围为 $[0,1]$ 的图像，PSNR 为：

$$
\operatorname{PSNR}(x,y)
=
10\log_{10}
\left(
\frac{1}{\operatorname{MSE}(x,y)}
\right).
$$

更一般地，若最大动态范围为 $MAX_I$：

$$
\operatorname{PSNR}(x,y)
=
10\log_{10}
\left(
\frac{MAX_I^2}{\operatorname{MSE}(x,y)}
\right).
$$

因此 PSNR 本质上是 image-space MSE 的对数变换。它能帮助发现：

- 曝光或颜色整体偏移；
- 背景或未编辑区域出现不必要变化；
- 输出有持续性的像素残差；
- 某个 loss 虽然让图像更锐利，却引入了更多逐像素错误。

但它和人类感知质量的一致性弱于 LPIPS。合理的光影变化、细小位置偏移和随机纹理
都会使 PSNR 明显下降；模糊的平均化结果反而可能获得较高 PSNR。

所以在当前实验中，PSNR 最合理的角色是：

$$
\boxed{\text{诊断项或防灾难性退化的 guard，而不是正向调参目标。}}
$$

---

## 50. KID 能补充什么，以及“无偏”为什么不等于“可靠”

KID 在 Inception 特征上计算 Maximum Mean Discrepancy：

$$
\operatorname{KID}
=
\operatorname{MMD}^2(F_{\mathrm{real}},F_{\mathrm{fake}}),
$$

通常使用三次多项式核：

$$
k(x,y)
=
\left(
\frac{x^\top y}{d}+1
\right)^3.
$$

对真实特征 $\{r_i\}_{i=1}^{m}$ 和生成特征
$\{g_j\}_{j=1}^{n}$，常用无偏估计为：

$$
\widehat{\operatorname{KID}}
=
\frac{1}{m(m-1)}
\sum_{i\ne i'} k(r_i,r_{i'})
+
\frac{1}{n(n-1)}
\sum_{j\ne j'} k(g_j,g_{j'})
-
\frac{2}{mn}
\sum_{i,j} k(r_i,g_j).
$$

KID 相比 FID 的一个优点是：这个有限样本 U-statistic 对固定特征表示是无偏估计。
但是：

$$
\boxed{\text{无偏} \ne \text{低方差}.}
$$

当前只有 10 张真实图和 10 张生成图：

- 每个集合内部只有 $\binom{10}{2}=45$ 个不同图像对；
- 跨集合只有 $10\times10=100$ 个图像对；
- 这些核项还不是相互独立的。

因此 KID 仍会有很大方差。由于使用无偏估计，10 张图时甚至可能得到轻微负数；
这不是代码错误，也不表示真实分布距离为负，而是有限样本噪声。

另一方面，当前 FID 使用 2048 维 Inception 特征，但每个集合只有 10 张图。经验
协方差的秩最多是：

$$
\operatorname{rank}(\widehat\Sigma)\le 10-1=9,
$$

远不足以稳定估计 2048 维协方差。因此当前的 FID 与 KID 都只能放在辅助列中，不能
依据很小的数值差异决定 loss 权重。

还有一个很直观的区别：

> 如果把 10 张生成图随机交换给另外 10 个 target，FID 和 KID 不会改变；  
> LPIPS、SSIM、PSNR、ArcFace 却会恶化。

这说明 FID/KID 只看两个集合，不检查输出是否对应正确的输入人物和目标图。

---

## 51. 在当前 `test_multi_gpu.py` 中如何正确实现

当前环境已经有：

- `torchmetrics==1.9.0`；
- `torch-fidelity==0.4.0`。

所以 SSIM、PSNR、KID 都可以直接实现，不需要安装新依赖。

### 51.1 SSIM 与 PSNR

当前 `image_to_tensor()` 返回 $[-1,1]$，这是 LPIPS 需要的范围。SSIM 和 PSNR
建议显式转成 $[0,1]$：

```python
from torchmetrics.functional.image import (
    peak_signal_noise_ratio,
    structural_similarity_index_measure,
)

image_01 = (image_tensor + 1.0) / 2.0
target_01 = (target_tensor + 1.0) / 2.0

ssim_score = structural_similarity_index_measure(
    image_01,
    target_01,
    data_range=1.0,
    reduction="none",
).mean().item()

psnr_score = peak_signal_noise_ratio(
    image_01,
    target_01,
    data_range=1.0,
    dim=(1, 2, 3),
    reduction="none",
).mean().item()
```

这里有几个不能忽略的细节：

1. 不要把 $[-1,1]$ 张量和 `data_range=1.0` 混用；
2. PSNR 要显式使用 `dim=(1, 2, 3)`，先得到每张图的 dB，再对样本求平均；
3. “逐图 PSNR 的均值”不等于“把所有像素合并后计算一次 PSNR”；
4. 多 GPU 进程应像当前 LPIPS 一样返回 `sum` 和 `count`，不能平均各 rank 的均值；
5. 生成图与 target 必须有相同尺寸，因此当前 resize 规则必须对所有模型保持固定。

### 51.2 KID

KID 可以复用 FID 当前使用的 RGB、`uint8`、$[0,255]$、NCHW 输入：

```python
from torchmetrics.image.kid import KernelInceptionDistance

kid = KernelInceptionDistance(
    feature=2048,
    subsets=1,
    subset_size=len(txt_files),
    normalize=False,
).to(device)
```

然后像 FID 一样：

```python
kid.update(target_uint8, real=True)
kid.update(image_uint8, real=False)
kid_mean, kid_std = kid.compute()
```

TorchMetrics 的默认值是：

```python
subset_size=1000
```

而当前每侧只有 10 张图，所以照搬默认值会直接报错。当前最诚实的临时设置是：

```python
subset_size=10
subsets=1
```

即使用全部 10 张图得到一个 full-sample KID。此时 `kid_std=0` 只因为只有一个
subset，**绝不表示 KID 很稳定，也不是置信区间**。

如果改成 `subsets=100, subset_size=5`，可以看到同一批 10 张图内部的随机子集
波动，但这些 subset 高度重叠，也不能当作模型不确定性。更合理的不确定性来自：

- 增加独立测试人物和场景；
- 固定多个推理 seed，报告跨 seed 波动；
- 对 LPIPS/ArcFace/SSIM/PSNR 做 paired bootstrap；
- 对最终候选使用多个独立训练 seed。

KID 必须在所有 worker 结束后，由主进程对完整集合计算。不能先在每张 GPU 上计算
局部 KID，再平均这些局部 KID。

---

## 52. 比新增指标更优先的评估修正

当前 `test_multi_gpu.py` 还有三个会直接影响 autoresearch 有效性的地方。

### 52.1 推理没有固定随机 seed

当前 pipeline 调用没有传入固定的 `generator`。如果基线与候选使用了不同噪声，
指标变化可能来自采样，而不是 loss。

必须为每个样本保存固定 seed，并保证它只由固定 manifest 决定，而不依赖：

- 当前 GPU rank；
- worker 数量；
- 样本被哪个进程分到；
- Python 的随机 hash。

也就是说，同一 `stem` 在所有 checkpoint、loss 组合和 GPU 数量下都必须使用同一
初始噪声。

### 52.2 ArcFace 需要同时报告检测成功率

当前代码直接取：

```python
face_app.get(image)[0]
```

没有检测到脸时会崩溃；多脸时也没有固定选择规则。因此至少应报告：

$$
\text{face detection rate}
=
\frac{\text{成功得到可比较人脸的样本数}}
{\text{总样本数}}.
$$

ArcFace 数值上升但检测成功率下降，不能判为更好。

### 52.3 mask 已读取但没有用于推理或评估

当前 `mask_image` 被读取，却没有进入 pipeline，也没有用于指标。若这个 mask 确实
表示编辑区域，可以进一步把指标分成：

- 编辑区域：LPIPS / SSIM，检查目标变化是否完成；
- 保持区域：LPIPS / SSIM / PSNR，检查人物、背景和非目标区域是否被破坏。

这通常比只增加一个全图指标更有解释力。

不过，不能简单把图像乘 mask 后直接计算普通 SSIM，因为 mask 边界和补零会污染
滑动窗口。更严谨的做法是使用 SSIM map 后在腐蚀过的 mask 内加权，或对稳定的
mask bounding box 单独计算。第一版仍可先加入全图 SSIM/PSNR，再做区域化扩展。

---

## 53. 三项训练 loss 的合理参数化

你的总损失构思是：

$$
\mathcal L_{\mathrm{total}}
=
\lambda_{\mathrm{MSE}}\mathcal L_{\mathrm{FM}}
+
\lambda_{\mathrm{cos}}\mathcal L_{\mathrm{cos}}
+
\lambda_{\mathrm{HF}}\mathcal L_{\mathrm{HF}}.
$$

三项分别约束：

| loss | 主要作用 |
|---|---|
| $\mathcal L_{\mathrm{FM}}$ | 逐元素速度误差，兼顾方向和模长 |
| $\mathcal L_{\mathrm{cos}}$ | 预测速度与正确速度的方向一致性 |
| $\mathcal L_{\mathrm{HF}}$ | 高频子带中的速度残差 |

这里有一个重要的调参问题：不建议同时自由搜索三个绝对权重。

若把三个权重同时乘以常数 $c$：

$$
(\lambda_{\mathrm{MSE}},\lambda_{\mathrm{cos}},\lambda_{\mathrm{HF}})
\mapsto
c(\lambda_{\mathrm{MSE}},\lambda_{\mathrm{cos}},\lambda_{\mathrm{HF}}),
$$

loss 的最优点不变，但梯度整体尺度改变，并与 learning rate、AdamW 状态和
gradient clipping 强耦合。这会让搜索同时混入“loss 比例”和“有效学习率”两个
问题。

因此应固定：

$$
\boxed{\lambda_{\mathrm{MSE}}=1,}
$$

只搜索两个辅助项相对于 FM-MSE 的强度。

这意味着形式上有三项 loss，但真正需要搜索的只有两个相对自由度。

---

## 54. 为什么不能直接给 cosine 和 HF 使用相同数值权重

三项原始 loss 的数值范围和梯度尺度完全不同：

- cosine loss 通常在一个有限范围内；
- FM-MSE 受 latent/velocity 尺度和 timestep weighting 影响；
- Haar 高频 residual 又受子带数量、归一化方式和 reduction 影响。

所以：

$$
\lambda_{\mathrm{cos}}=0.1,\qquad
\lambda_{\mathrm{HF}}=0.1
$$

并不表示两者对 LoRA 参数产生同样强的训练影响。

在搜索前，应在一组固定 calibration batches、固定噪声、固定 timestep 上测量
三个 loss 对 LoRA 参数的梯度范数：

$$
G_k
=
\operatorname{median}
\left\|
\nabla_{\theta_{\mathrm{LoRA}}}\mathcal L_k
\right\|_2,
\qquad
k\in\{\mathrm{FM},\mathrm{cos},\mathrm{HF}\}.
$$

然后冻结归一化：

$$
\widetilde{\mathcal L}_{\mathrm{cos}}
=
\frac{G_{\mathrm{FM}}}{G_{\mathrm{cos}}+\epsilon}
\mathcal L_{\mathrm{cos}},
$$

$$
\widetilde{\mathcal L}_{\mathrm{HF}}
=
\frac{G_{\mathrm{FM}}}{G_{\mathrm{HF}}+\epsilon}
\mathcal L_{\mathrm{HF}}.
$$

注意：这个归一化尺度只能在搜索开始前计算一次，并对所有候选冻结。不能让每个
候选重新计算，否则每个实验的目标函数又发生了变化。

---

## 55. 更适合 autoresearch 的二维搜索坐标

经过梯度归一化后，总 loss 可以写成：

$$
\boxed{
\mathcal L
=
\mathcal L_{\mathrm{FM}}
+
\rho
\left[
\pi\widetilde{\mathcal L}_{\mathrm{cos}}
+
(1-\pi)\widetilde{\mathcal L}_{\mathrm{HF}}
\right].
}
$$

其中：

- $\rho$：辅助 loss 的总强度；
- $\pi$：辅助强度在 cosine 与 HF 之间的分配；
- $\rho=0$：纯 FM-MSE 基线；
- $\pi=1$：cosine-only；
- $\pi=0$：HF-only；
- $0<\pi<1$：两项组合。

第一轮候选可以使用：

$$
\rho\in\{0,\ 0.03,\ 0.10,\ 0.30\},
$$

$$
\pi\in\{0,\ 0.25,\ 0.5,\ 0.75,\ 1\}.
$$

$\rho=0$ 只计算一次，因此总共是：

$$
1+3\times5=16
$$

个候选，而不是 20 个。

这组数值只是以“归一化后的辅助梯度占 FM 梯度的小比例”为起点。若最优点落在
$\rho=0.30$ 的边界，第二轮才向更大强度局部扩展；若最优点是 $\rho=0$，说明当前
数据和训练预算下辅助 loss 没有提供可验证收益，不应强行组合。

---

## 56. autoresearch 不应直接最小化六项指标的加权和

不建议定义：

$$
a\,\mathrm{LPIPS}
-b\,\mathrm{ArcFace}
-c\,\mathrm{SSIM}
-d\,\mathrm{PSNR}
+e\,\mathrm{FID}
+f\,\mathrm{KID}.
$$

原因包括：

1. 六项量纲不同；
2. LPIPS、SSIM、PSNR 之间部分重复；
3. FID 与 KID 都是 Inception 集合指标，也部分重复；
4. FID/KID 在 10 张图上噪声很大；
5. 加权和允许一个指标的巨大改善掩盖另一个核心指标的明显退化；
6. 搜索器可能学会“少做必要编辑”，以换取较高 PSNR、SSIM 或 ArcFace。

更合理的是：

- LPIPS 与 ArcFace 共同决定主排名；
- face detection rate 是硬 guard；
- SSIM 是结构退化 guard；
- PSNR 是宽松诊断 guard；
- FID/KID 只记录，不参与当前自动接受判定。

---

## 57. LPIPS 与 ArcFace 的平衡 autoresearch 分数

对完全相同的图像、prompt 和推理 seed，定义候选相对基线的 paired 差值：

$$
d_L
=
\mathrm{LPIPS}_{c}
-
\mathrm{LPIPS}_{b},
$$

$$
d_A
=
\mathrm{ArcFace}_{c}
-
\mathrm{ArcFace}_{b}.
$$

其中：

- $d_L<0$ 表示 LPIPS 改善；
- $d_A>0$ 表示 ArcFace 改善。

先通过重复运行基线估计自然波动，并冻结两个“实质变化尺度”：

$$
\tau_L
=
\max(\delta_L^{\mathrm{practical}},2SE_L),
$$

$$
\tau_A
=
\max(\delta_A^{\mathrm{practical}},2SE_A).
$$

若还没有经验阈值，可以把：

$$
\delta_L^{\mathrm{practical}}=0.005,
\qquad
\delta_A^{\mathrm{practical}}=0.01
$$

作为第一版起点，但必须在正式启动 autoresearch 前固定，不能看完候选结果后再改。

通过 paired bootstrap 得到置信区间，定义：

$$
q_L
=
\operatorname{clip}
\left(
\frac{-UCB_{95}(d_L)}{\tau_L},
-3,3
\right),
$$

$$
q_A
=
\operatorname{clip}
\left(
\frac{LCB_{95}(d_A)}{\tau_A},
-3,3
\right).
$$

直观理解：

- $q_L>0$：即使考虑不确定性，LPIPS 仍有改善证据；
- $q_A>0$：即使考虑不确定性，ArcFace 仍有改善证据；
- 负数：相应指标有退化风险。

给 autoresearch 一个单一机械分数：

$$
\boxed{
S
=
\frac{q_L+q_A}{2}
-
\frac{|q_L-q_A|}{4}.
}
$$

第二项惩罚两项改善非常不平衡的候选。例如“LPIPS 大幅改善但 ArcFace 明显下降”
不会轻易成为 incumbent。

建议的机械接受条件为：

```text
guard_passed
AND score >= incumbent_score + 0.25
AND (q_lpips > 0 OR q_arcface > 0)
```

同时保留 Pareto archive：只要一个候选在 LPIPS 和 ArcFace 上都没有实质退化，并
至少有一项实质改善，就可以被记录为非劣候选；标量 $S$ 只是为了满足 autoresearch
每轮必须选择一个 incumbent 的需要。

训练 loss 里的 cosine 和评估 ArcFace cosine 虽然都使用余弦形式，但它们位于完全
不同的空间：

- 前者比较 latent velocity 的方向；
- 后者比较人脸识别 embedding 的方向。

所以不能把二者当成同一个目标，也不能预设加入 velocity cosine loss 后 ArcFace
一定上升。

---

## 58. 推荐的 guard

在正式搜索前冻结以下 guard：

1. 训练与评估无 NaN、无崩溃；
2. face detection rate 不低于基线；
3. LPIPS 不允许劣于基线超过 $\tau_L$；
4. ArcFace 不允许劣于基线超过 $\tau_A$；
5. SSIM 不允许出现超过预设阈值的结构退化；
6. PSNR 只设置宽松的灾难性退化阈值，例如先从 $0.5\ \mathrm{dB}$ 开始校准；
7. 所有候选使用相同推理 seed、steps、guidance、checkpoint 选择规则；
8. `test_multi_gpu.py` 在搜索期间冻结，不能让 autoresearch 修改评估器来提高分数。

当前阶段不应把 FID 或 KID 放入硬 guard，因为 10 张图下它们可能因为抽样噪声拒绝
真正更好的模型。

---

## 59. 昂贵训练下的 successive-halving 搜索

直接把 16 个候选都完整训练多个 seed 成本很高。可以采用分阶段淘汰：

| 阶段 | 候选数 | 训练步数 | dev 图像 / 推理 seed | 作用 |
|---|---:|---:|---|---|
| Round 0 | 基线 | 多次 | 固定全集 | 测量噪声、冻结阈值与归一化 |
| Rung 1 | 16 | 50 | 4 个 pair / 1 seed | 只淘汰崩溃和明显退化 |
| Rung 2 | 前 6 | 100 | 8–10 个 pair / 2 seeds | 初步可靠排名 |
| Rung 3 | 前 3 | 150/200 | 完整 dev / 3 seeds | 选 finalist |
| Final | 前 2 + MSE baseline | 完整 | sealed holdout / 至少 3 training seeds | 最终验证 |

前三个 training rung 的新增训练步数约为：

$$
16\times50
+
6\times50
+
3\times100
=
1400,
$$

大致相当于 7 次完整的 200-step run，再加 finalist 的多训练 seed 成本。

历史 checkpoint 结果提示 100 steps 可能优于 200 steps。因此所有候选都应保存
50/100/150/200 checkpoints，并使用完全相同的 dev 早停规则。不能允许某个候选
比其他候选拥有更多挑 checkpoint 的机会，否则 checkpoint selection 本身会成为
隐藏超参数。

---

## 60. dev 与 sealed holdout 必须分开

当前 10 张测试图不能一边反复指导 loss 权重搜索，一边又作为论文最终 test 结果。
否则 autoresearch 会对这 10 张图过拟合。

推荐的数据角色是：

1. 当前 10 张可以暂时作为 dev；
2. 另建 sealed final holdout，至少 20–50 个 paired 样本；
3. 若要让 FID 承担正式结论，最终集合应进一步扩大到数百；
4. 按人物身份、原始拍摄序列或近重复场景分组切分，避免泄漏；
5. 固定并保存 manifest、prompt、target、生成 seed、评估器版本与 pipeline 参数；
6. autoresearch 循环不能看到 sealed holdout 分数；
7. 只有 top-2、纯 MSE baseline 可以各调用一次 holdout；
8. 若 holdout 排名反转，应报告结果不确定，不能回到同一 holdout 上继续调权重。

如果暂时无法建立新 holdout，那么可以做探索性搜索，但结果只能表述为：

> 在当前 10 个 dev pairs 上观察到改善。

不能把它直接表述为模型泛化能力已经提高。

---

## 61. 建议的最终评估表

主表可以写成：

| Method | LPIPS $\downarrow$ | ArcFace $\uparrow$ | Face Det. $\uparrow$ | SSIM $\uparrow$ | PSNR $\uparrow$ | FID $\downarrow$ | KID $\downarrow$ |
|---|---:|---:|---:|---:|---:|---:|---:|
| FM-MSE baseline |  |  |  |  |  |  |  |
| + Cosine |  |  |  |  |  |  |  |
| + HF |  |  |  |  |  |  |  |
| + Cosine + HF |  |  |  |  |  |  |  |

表注应明确：

- LPIPS、ArcFace 是 primary metrics；
- SSIM 是 structural auxiliary metric；
- PSNR 是 pixel-fidelity diagnostic；
- FID/KID 在当前小样本下是 exploratory distributional metrics；
- PSNR 单位为 dB；
- KID 报告的是 raw KID 还是经过 $\times100$ / $\times1000$ 的展示值；
- 所有方法使用相同的生成 seeds；
- Face Det. 应写成比例或 `成功数/总数`。

如果空间允许，再提供 per-sample 表或箱线图；只有 10 个样本时，仅报告平均值很容易
被一两个异常样本主导。

---

## 62. 本轮建议的行动顺序

最合理的顺序是：

$$
\boxed{
\begin{aligned}
&\text{固定逐样本推理 seed、ArcFace 检测规则}\\
\rightarrow\;&\text{加入 SSIM、PSNR、KID，并输出机器可读 metrics JSON}\\
\rightarrow\;&\text{实现并单测 FM + cosine + HF 三项 loss}\\
\rightarrow\;&\text{测固定 calibration batches 的分项 loss / 梯度范数}\\
\rightarrow\;&\text{重复基线，冻结 autoresearch score 与 guards}\\
\rightarrow\;&\text{按 }(\rho,\pi)\text{ 做 successive halving}\\
\rightarrow\;&\text{top-2 + baseline 在 sealed holdout、多训练 seed 上验证}.
\end{aligned}
}
$$

这套设计保留 FID 和 KID 来丰富评估表，又不会让它们在 10 张图上误导
autoresearch；同时把 SSIM 和 PSNR 放到它们最擅长的位置。

---

## 63. 本节参考资料

- [ModaFlow: Modality-Aware Flow Matching for High-Fidelity Virtual Try-On](https://arxiv.org/abs/2606.27773)
- [Image Quality Assessment: From Error Visibility to Structural Similarity（SSIM）](https://doi.org/10.1109/TIP.2003.819861)
- [The Unreasonable Effectiveness of Deep Features as a Perceptual Metric（LPIPS）](https://openaccess.thecvf.com/content_cvpr_2018/html/Zhang_The_Unreasonable_Effectiveness_of_Deep_CVPR_2018_paper.html)
- [Demystifying MMD GANs（KID）](https://arxiv.org/abs/1801.01401)
- [Effectively Unbiased FID and Inception Score and Where to Find Them](https://openaccess.thecvf.com/content_CVPR_2020/html/Chong_Effectively_Unbiased_FID_and_Inception_Score_and_Where_to_Find_CVPR_2020_paper.html)
- [TorchMetrics 1.9.0: SSIM](https://lightning.ai/docs/torchmetrics/stable/image/structural_similarity.html)
- [TorchMetrics 1.9.0: PSNR](https://lightning.ai/docs/torchmetrics/stable/image/peak_signal_noise_ratio.html)
- [TorchMetrics 1.9.0: KID](https://lightning.ai/docs/torchmetrics/stable/image/kernel_inception_distance.html)
