---
name: kaggle-tpu-accelerate-guide
description: "Kaggle TPU v5e-8 + Accelerate 分布式训练实战指南。涵盖环境配置、核心 API、常见陷阱及解决方案。Triggers: TPU训练, Kaggle TPU, Accelerate, 分布式训练, TPU补丁, TPU环境配置。"
---

# Kaggle TPU v5e-8 + Accelerate 分布式训练实战指南

> 本文总结了在 Kaggle TPU v5e-8 上使用 PyTorch + Accelerate 进行分布式深度学习训练的完整经验。涵盖环境配置、核心 API、常见陷阱及解决方案。适用于任意模型架构（Transformer、CNN、MLP 等）。

---

## 1. 环境准备

### 1.1 安装依赖

Kaggle TPU VM 预装了 PyTorch 和 torch_xla，但需要额外安装 accelerate：

```python
!pip install accelerate tiktoken -q
```

**关键：必须卸载 tensorflow 并换为 tensorflow-cpu**。Kaggle TPU 环境默认安装的 tensorflow 会占用 TPU 资源，与 PyTorch XLA 冲突：

```python
!pip uninstall -y -q tensorflow
!pip install -q tensorflow-cpu
```

### 1.2 TPU 环境补丁（必须最先执行）

这段代码**必须在任何 `import torch` 之前执行**，否则会报错或行为异常：

```python
import os

# 告诉 XLA 使用 TPU 作为设备
os.environ["PJRT_DEVICE"] = "TPU"

# 限制 CPU 线程数，避免多进程竞争
os.environ["OMP_NUM_THREADS"] = "1"

# 删除这两个环境变量，它们会干扰 accelerate 的多进程启动
if "TPU_PROCESS_ADDRESSES" in os.environ:
    del os.environ["TPU_PROCESS_ADDRESSES"]
if "CLOUD_TPU_TASK_ID" in os.environ:
    del os.environ["CLOUD_TPU_TASK_ID"]

# TPU 内存优化：增大 tensor 分配器上限
os.environ['XLA_TENSOR_ALLOCATOR_MAXSIZE'] = '100000000'
```

**为什么需要这些？**

| 环境变量 | 作用 | 不设会怎样 |
|---------|------|-----------|
| `PJRT_DEVICE=TPU` | 告诉 PyTorch/XLA 使用 PJRT 运行时连接 TPU | 报错找不到设备 |
| `OMP_NUM_THREADS=1` | 限制 OpenMP 线程数 | 8 个进程各抢 CPU 线程，性能下降甚至死锁 |
| 删除 `TPU_PROCESS_ADDRESSES` | 避免 XLA 尝试连接不存在的多节点 TPU | 启动报错 |
| 删除 `CLOUD_TPU_TASK_ID` | 清除遗留的多节点标识 | 进程标识混乱 |
| `XLA_TENSOR_ALLOCATOR_MAXSIZE` | 增大 XLA 张量分配器缓存上限 | 大张量分配失败或频繁重新分配 |

### 1.3 Kaggle Notebook 启动方式

```python
from accelerate import notebook_launcher

def train_worker():
    # 所有训练逻辑写在这里
    # 每个 TPU core 运行一个此函数的实例
    ...

# 启动 8 个进程，对应 TPU v5e-8 的 8 个 core
notebook_launcher(train_worker, num_processes=8)
```

`notebook_launcher` 会 fork 出 8 个进程，每个进程绑定一个 TPU core。所有进程执行同一个 `train_worker` 函数，通过 `accelerator.process_index` 区分身份。

---

## 2. 核心概念速览

### TPU v5e-8 的硬件模型

```
┌─────────────────────────────────────────────┐
│              Kaggle TPU v5e-8               │
│                                             │
│   Core 0   Core 1   Core 2   Core 3        │
│   Core 4   Core 5   Core 6   Core 7        │
│                                             │
│   每个 core 有独立的 HBM（约 16GB）           │
│   core 间通过 ICI 高速互联（all-reduce 等）   │
└─────────────────────────────────────────────┘
```

- **8 个 core = 8 个进程**，每个进程独立运行模型的完整副本
- 每个进程处理不同的数据 batch（数据并行）
- 梯度通过 all-reduce 在 8 个进程间同步
- Accelerate 帮你封装了这些分布式细节

### Accelerate 的角色

Accelerate 是 HuggingFace 出品的分布式训练封装库，它的核心价值：

1. **统一 API**：同一套代码适配单 GPU、多 GPU、TPU、多节点
2. **自动处理分布式细节**：模型包装、梯度同步、混合精度、checkpoint 保存/加载
3. **`notebook_launcher`**：唯一靠谱的在 Kaggle Notebook 里启动 TPU 多进程的方式

---

## 3. Accelerate 核心 API 详解

### 3.1 创建 Accelerator

```python
from accelerate import Accelerator

accelerator = Accelerator(
    mixed_precision="bf16",              # TPU 支持 bf16，不支持 fp16
    gradient_accumulation_steps=4,       # 梯度累积步数，必须在构造时传入
)
```

**重要：`gradient_accumulation_steps` 必须在构造时指定**，不能在训练过程中修改。`accumulate()` 上下文管理器依赖此值决定何时做梯度同步。

### 3.2 准备模型和优化器

```python
model, optimizer = accelerator.prepare(model, optimizer)
```

`prepare()` 会：
- 将模型包装为分布式模型（DDP 或 XLA 分布式）
- 确保优化器状态在多进程间正确同步
- 在 TPU 上，模型参数会被移到 XLA 设备

**顺序很重要**：先创建模型和优化器，再 `prepare()`。

### 3.3 前向 / 反向 / 优化器步进

```python
with accelerator.autocast():          # 自动混合精度上下文
    logits, loss = model(x, y)

accelerator.backward(loss)             # 替代 loss.backward()
accelerator.clip_grad_norm_(model.parameters(), 1.0)  # 梯度裁剪
optimizer.step()
```

### 3.4 数据移到设备

```python
x = x.to(accelerator.device)   # 自动移到当前进程的 TPU core
y = y.to(accelerator.device)
```

### 3.5 多进程同步

```python
accelerator.wait_for_everyone()   # 阻塞直到所有进程都到达此处
```

用于在保存 checkpoint、采样等操作前确保所有进程同步。

### 3.6 收集各进程的数据

```python
# 收集所有进程的 loss 值，求全局平均
local_loss = loss.detach()
global_loss = accelerator.gather(local_loss).mean().item()
```

### 3.7 获取原始模型

```python
raw_model = accelerator.unwrap_model(model)
# 用于访问原始模型的属性（如 config），或保存 state_dict
```

### 3.8 保存 Checkpoint

```python
accelerator.save(ckpt, save_path)   # 安全的多进程保存
```

**不要用 `torch.save()`**，在 TPU 上可能引发多进程互锁。`accelerator.save()` 内部会处理 XLA 张量到 CPU 的转换。

### 3.9 设置随机种子

```python
from accelerate.utils import set_seed

# 每个进程用不同的种子，保证数据 shuffle 不同但可复现
process_seed = 1337 + accelerator.process_index
set_seed(process_seed)
```

---

## 4. TPU 上的 bf16 混合精度

### 4.1 为什么用 bf16 而不是 fp16？

TPU 原生支持 bf16（Brain Float 16），不支持 fp16 的快速运算。bf16 的特点是：
- 与 fp32 相同的指数位（8 位），动态范围大，不容易溢出
- 尾数位较少（7 位 vs fp32 的 23 位），精度略低

**正确开启 bf16 的方式只有一种**：用 Accelerate 的 `mixed_precision="bf16"`（autocast 混合精度）。**绝对不要**用 `XLA_USE_BF16=1` 全局开关，它是导致训练 NaN 的头号元凶，详见 [4.4 节](#44-永远不要用-xla_use_bf16-全局开关实战教训)。

### 4.2 必须强制 fp32 的关键路径

bf16 精度不够的地方会产生 NaN。以下路径**必须手动转 fp32**：

#### LayerNorm

```python
# ❌ 错误：bf16 下方差计算会 catastrophic cancellation → NaN
x = self.ln_1(x)

# ✅ 正确：先转 fp32，归一化后再转回
x = self.ln_1(x.float()).to(x.dtype)
```

原因：LayerNorm 内部计算方差时需要 `E[x²] - E[x]²`，两个相近的大数相减，bf16 精度不够会导致结果为 0 或负数，开方后 NaN。

#### Cross-Entropy Loss

```python
# ❌ 错误：bf16 下 log_softmax 会溢出为 -inf → NaN
loss = F.cross_entropy(logits.view(-1, vocab_size), targets.view(-1))

# ✅ 正确：logits 转 fp32 再算
loss = F.cross_entropy(logits.float().view(-1, vocab_size), targets.view(-1))
```

### 4.3 QK-Norm：可选的防御手段（非必须）

QK-Norm 能抑制 attention logit 数值过大，但**它不是训练稳定性的必需品**。实测证明：只要混合精度姿势正确（Accelerate autocast + 4.2 节的关键路径 fp32 保护），即使完全去掉 QK-Norm，1.23 亿参数规模的模型也能稳定训练上万步不出 NaN。

它的真实定位是**防御性保险**：对抗训练过程中 attention logits 缓慢增大的现象（logit drift，已知现象），在低精度环境下能更早避免 softmax 溢出。建议：

- 想简化模型 → 可以去掉，但**必须保留 4.2 节的 fp32 保护**
- 追求稳健（尤其是更大模型、更长训练）→ 建议保留：

```python
self.q_norm = nn.LayerNorm(head_dim)
self.k_norm = nn.LayerNorm(head_dim)

# 在计算 attention score 之前归一化 Q 和 K
q = self.q_norm(q.float()).to(q.dtype)
k = self.k_norm(k.float()).to(k.dtype)
```

### 4.4 永远不要用 XLA_USE_BF16 全局开关（实战教训）

```python
# ❌ 致命：强制所有浮点张量为 bf16（包括参数、梯度、优化器状态）
os.environ['XLA_USE_BF16'] = '1'

# ✅ 正确：用 Accelerate autocast 开启混合精度（fp32 主权重 + bf16 计算）
accelerator = Accelerator(mixed_precision="bf16")
```

两者的本质区别：

| 模式 | `XLA_USE_BF16=1`（❌） | Accelerate autocast（✅） |
|------|----------------------|--------------------------|
| 参数 / 梯度 | bf16 | fp32（master weights） |
| 优化器状态（AdamW 动量） | **bf16，精度严重受损** | fp32 |
| 手动 `.float()` 保护 | 被失效（强制降回 bf16） | 有效 |
| 结果 | 收敛变慢、误差累积、NaN 崩溃 | 数值稳定 |

**实战教训**：本项目早期复现时设置了 `XLA_USE_BF16=1` 且敏感路径无 fp32 保护，训练严重退化（6000 步 loss 仍停留在 ~6.3，而正确姿势同期约 3.5），并在第 6056 步突然全面 NaN 崩溃。根因：AdamW 优化器状态被降级为 bf16 后，动量累积的舍入误差使自适应学习率逐渐失真；同时 LayerNorm 方差、cross_entropy 的 log_softmax、softmax 等敏感路径毫无保护，误差累积到某一步单点溢出，NaN 经梯度污染全部参数。
**记住**：混合精度的“混合”靠的是 fp32 主权重和 fp32 优化器状态；`XLA_USE_BF16=1` 会把“混合精度”变成“全链路低精度”。

---

## 5. XLA 计算图与同步机制

### 5.1 XLA 的惰性执行模型

这是 TPU 编程与 GPU 最大的区别：

- **GPU**：CUDA 操作基本是立即执行的
- **TPU (XLA)**：操作被记录到一个**计算图**中，只有在 `sync()` 时才真正执行

```python
import torch_xla
import torch_xla.core.xla_model as xm

a = torch.tensor([1.0], device="xla:0")
b = a + 1   # 此时 b 还没真正计算！只是一个图节点
c = b * 2   # 同上

torch_xla.sync()  # 现在 a→b→c 的计算图被编译并执行
print(c)          # 现在才能拿到真实值
```

### 5.2 什么时候必须 sync？

#### 训练循环末尾

```python
# 每个 optimizer step 结束后同步
optimizer.step()
optimizer.zero_grad()

if use_xla_sync:
    torch_xla.sync()
```

#### 验证循环中每步之后

```python
for _ in range(val_steps):
    _, loss = model(x, y)

    # ❌ 危险：不 sync 的话，20 步的计算图会累积
    # logits (B, T, vocab_size) ≈ 3.1GB/步 × 20 步 → OOM
    losses.append(loss.detach())

    # ✅ 正确：每步立即同步，释放中间张量
    if use_xla_sync:
        torch_xla.sync()
    losses.append(loss.detach())
```

#### 从 CPU 加载权重到 TPU 后

```python
model, optimizer = accelerator.prepare(model, optimizer)

# 强制同步：确保从 CPU 加载的权重真正物化到 TPU 内存
if use_xla_sync:
    xm.mark_step()
```

### 5.3 `sync()` vs `mark_step()`

- `torch_xla.sync()`：触发计算图执行，更常用
- `xm.mark_step()`：标记一个计算图的边界，通常用于训练循环中分割计算图

在实践中两者都可以用于强制同步，`sync()` 语义更明确。

---

## 6. 梯度累积

### 6.1 为什么要梯度累积？

TPU 每个 core 的 HBM 有限（约 16GB），大 batch 可能放不下。梯度累积允许用小 batch 多次前向-反向，累积梯度后再做一次 optimizer step，等效于大 batch。

```
等效 batch size = micro_batch_size × seq_len × grad_accum_steps × num_processes
524288 = 32 × 1024 × 2 × 8
```

### 6.2 Accelerate 的 accumulate 上下文管理器

```python
grad_accum_steps = TOTAL_BATCH_SIZE // (MICRO_BATCH_SIZE * SEQ_LEN * world_size)

# 构造 Accelerator 时传入
accelerator = Accelerator(
    gradient_accumulation_steps=grad_accum_steps,
)

# 训练循环中使用
with accelerator.accumulate(model):
    for micro_step in range(grad_accum_steps):
        x, y = train_loader.next_batch()
        x = x.to(device)
        y = y.to(device)

        with accelerator.autocast():
            _, loss = model(x, y)

        accelerator.backward(loss)

    # accumulate 上下文会自动处理：
    # - 前 N-1 步：跳过梯度同步（all-reduce）
    # - 最后一步：自动做梯度同步
    # 比手动调用少了 N-1 次 all-reduce 通信

    optimizer.step()
    optimizer.zero_grad()
```

### 6.3 手动梯度累积 vs accumulate 上下文

```python
# ❌ 手动方式：每次 backward 都会触发 all-reduce，通信开销大
for micro_step in range(grad_accum_steps):
    loss = model(x, y)
    (loss / grad_accum_steps).backward()   # 每步都 all-reduce

# ✅ accumulate 上下文：只在最后一步 all-reduce
with accelerator.accumulate(model):
    for micro_step in range(grad_accum_steps):
        loss = model(x, y)
        accelerator.backward(loss)          # 前 N-1 步跳过同步
```

---

## 7. 梯度检查点（Gradient Checkpointing）

### 7.1 原理

正常前向传播会保存所有中间激活值用于反向传播。梯度检查点只保存每层的输入，反向传播时重新计算中间值。用计算时间换内存。

### 7.2 XLA 版本的 checkpoint

TPU 上必须用 `torch_xla.utils.checkpoint`，不能直接用 PyTorch 的版本：

```python
try:
    from torch_xla.utils.checkpoint import checkpoint as xla_checkpoint
    HAS_XLA_CHECKPOINT = True
except ImportError:
    HAS_XLA_CHECKPOINT = False

# 在模型 forward 中使用（以 Transformer block 列表为例）
for block in self.layers:
    if self.gradient_checkpointing and self.training:
        if x.device.type == "xla" and HAS_XLA_CHECKPOINT:
            x = xla_checkpoint(block, x)          # XLA 版本
        else:
            x = torch.utils.checkpoint.checkpoint(  # PyTorch 版本（GPU 回退）
                block, x, use_reentrant=False
            )
    else:
        x = block(x)
```

**为什么不能用 PyTorch 原版？** PyTorch 的 checkpoint 实现依赖 CUDA 特定的 autograd hook，在 XLA 设备上会报错或行为异常。XLA 版本做了适配。

### 7.3 开启方式

```python
model.gradient_checkpointing = True
```

建议在 `accelerator.prepare()` 之前设置。

---

## 8. 模型设计注意事项

### 8.1 Weight Tying 被 prepare() 破坏（Transformer 模型通用问题）

这是 TPU + Accelerate 最隐蔽的坑之一。

很多 Transformer 模型（GPT、LLaMA、BERT 等）会把输入 embedding 和输出投影层的权重绑定（weight tying），即 `lm_head.weight = wte.weight`。但在 TPU 上：

```python
# ❌ 问题：accelerator.prepare() 会破坏 weight tying
model = MyTransformer(config)   # lm_head.weight 和 wte.weight 指向同一块内存
model, optimizer = accelerator.prepare(model, optimizer)
# 此时 lm_head.weight 和 wte.weight 可能已经是不同的张量了！
# prepare() 内部会为分布式训练重新组织参数，weight tying 的引用关系丢失
```

**解决方案：不用 lm_head，直接用 F.linear**

```python
class MyTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.layers = nn.ModuleList([MyBlock(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        # 故意不定义 lm_head！

    def forward(self, idx, targets=None):
        x = self.embedding(idx)
        # ... transformer blocks ...
        x = self.ln_f(x.float()).to(x.dtype)

        # 直接用 F.linear，手动传入 embedding.weight 作为转置权重
        # 这样无论 prepare() 怎么处理参数，都能保证是同一份权重
        logits = F.linear(x, self.embedding.weight)
        return logits, loss
```

### 8.2 SDPA（Scaled Dot-Product Attention）在 TPU 上不可用

PyTorch 的 `F.scaled_dot_product_attention` 是为 CUDA 优化的，在 XLA 设备上要么不支持，要么走很慢的 fallback 路径。

```python
# 需要检测设备类型，选择不同的 attention 实现
if self.use_sdpa and x.device.type != "xla":
    # GPU 上走高效的 fused attention
    y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
else:
    # TPU 上走手写 attention
    att = (q * self.scale) @ k.transpose(-2, -1)
    att = att.masked_fill(~mask_slice, -1e4)
    att = F.softmax(att, dim=-1)
    y = att @ v
```

### 8.3 torch.compile 在 TPU 上不要用

```python
# ❌ TPU 上 torch.compile 几乎没有加速效果，还可能引入 bug
model = torch.compile(model)

# ✅ 直接用原生模型
model = model
```

XLA 本身就是一个编译器（将 PyTorch 操作编译为 TPU 可执行的 HLO IR），再套一层 `torch.compile` 没有意义。

### 8.4 残差投影层零初始化（可选，通用技巧）

对于使用残差连接的深层网络（如 Transformer），可以将残差分支的最后一层初始化为零，使得训练初期残差流为恒等映射，有助于稳定训练：

```python
# 用自定义属性标记需要零初始化的层
class MyBlock(nn.Module):
    def __init__(self, config):
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.ZERO_INIT = 1   # 标记为需要零初始化

def _init_weights(self, module):
    if isinstance(module, nn.Linear):
        if hasattr(module, "ZERO_INIT"):
            torch.nn.init.zeros_(module.weight)  # 残差流初期为恒等映射
        else:
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
```

---

## 9. Checkpoint 保存与加载

### 9.1 保存

```python
def save_checkpoint(accelerator, model, optimizer, train_loader, val_loader,
                    step, best_val_loss, model_config, save_path):
    # 1. 等待所有进程同步
    accelerator.wait_for_everyone()

    # 2. 获取去除分布式包装后的 state_dict
    state_dict = accelerator.get_state_dict(model)

    # 3. 构建 checkpoint 字典
    ckpt = {
        "model": state_dict,
        "optimizer": optimizer.state_dict(),
        "step": step,
        "best_val_loss": best_val_loss,
        "model_config": asdict(model_config),
        "train_loader_state": train_loader.state_dict(),
        "val_loader_state": val_loader.state_dict(),
    }

    # 4. 安全保存（处理 XLA 张量到 CPU 的转换）
    accelerator.save(ckpt, save_path)

    # 5. 再次等待同步
    accelerator.wait_for_everyone()
```

**绝对不要手动 `.cpu()` 转换 XLA 张量**，会引发多进程互锁死锁。用 `accelerator.save()` 统一处理。

### 9.2 加载

```python
# 1. 在 CPU 上加载 checkpoint
checkpoint = torch.load(RESUME_PATH, map_location="cpu")

# 2. 创建模型，加载权重（在 prepare() 之前）
model = MyModel(model_config)
state_dict = checkpoint["model"]

# 3. 清理可能的前缀
for prefix in ['_orig_mod.', 'module.']:
    for k in list(state_dict.keys()):
        if k.startswith(prefix):
            state_dict[k[len(prefix):]] = state_dict.pop(k)

model.load_state_dict(state_dict, strict=False)

# 4. 确保参数为 fp32（checkpoint 可能是 bf16 保存的）
for name, param in model.named_parameters():
    if param.dtype != torch.float32:
        param.data = param.data.float()

# 5. 创建优化器
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

# 6. prepare() 之后再加载 optimizer state
model, optimizer = accelerator.prepare(model, optimizer)
if use_xla_sync:
    xm.mark_step()   # 确保模型权重物化到 TPU

# 7. optimizer state 必须在 prepare() 之后加载
optimizer.load_state_dict(checkpoint["optimizer"])
if use_xla_sync:
    xm.mark_step()
```

### 9.3 加载顺序总结

```
torch.load (CPU) → 创建模型 → load_state_dict → 创建优化器
→ accelerator.prepare(model, optimizer) → xm.mark_step()
→ optimizer.load_state_dict → xm.mark_step()
```

**optimizer state 必须在 prepare() 之后加载**，因为 prepare() 会改变优化器的内部结构。

---

## 10. 多进程同步陷阱大全

### 陷阱 1：evaluate 中不同步导致 OOM

```python
# ❌ 错误：多步的计算图累积在内存中，中间张量无法释放 → OOM
for _ in range(20):
    _, loss = model(x, y)
    losses.append(loss.detach())

# ✅ 正确：每步同步，立即释放中间张量
for _ in range(20):
    _, loss = model(x, y)
    torch_xla.sync()           # 立即执行计算图，释放中间张量
    losses.append(loss.detach())
```

### 陷阱 2：部分进程执行导致死锁

```python
# ❌ 致命：只有 rank 0 做推理，其他 7 个 core 空闲等待
# XLA 的 all-reduce 需要所有 core 参与，否则永久死锁！
if accelerator.process_index == 0:
    run_evaluation(model)

# ✅ 正确：所有 8 个 core 都参与前向计算
run_evaluation(model)   # 所有进程都执行，只有 rank 0 打印/保存结果

# 在函数内部：
def run_evaluation(model, accelerator, ...):
    # 所有进程都做推理（XLA 要求计算图同步）
    outputs = model(eval_data)
    ...

    # 只有 rank 0 打印/保存结果
    if accelerator.process_index == 0:
        print(results)
```

### 陷阱 3：保存 checkpoint 时不同步

```python
# ❌ 错误：只有 rank 0 保存，其他进程不等待
if accelerator.process_index == 0:
    torch.save(ckpt, path)   # 其他进程可能还在计算 → 竞态

# ✅ 正确：保存前后都等待
accelerator.wait_for_everyone()
accelerator.save(ckpt, path)
accelerator.wait_for_everyone()
```

### 陷阱 4：手动 .cpu() 转换 XLA 张量

```python
# ❌ 危险：手动 .cpu() 在多进程环境下会互锁死锁
state_dict = {k: v.cpu() for k, v in model.state_dict().items()}

# ✅ 正确：用 accelerator.save() 自动处理
accelerator.save(ckpt, path)
```

### 陷阱 5：optimizer state 加载时机错误

```python
# ❌ 错误：在 prepare() 之前加载 optimizer state
optimizer.load_state_dict(checkpoint["optimizer"])   # 此时 optimizer 还没被包装
model, optimizer = accelerator.prepare(model, optimizer)  # prepare 会重置 optimizer 结构！

# ✅ 正确：prepare() 之后再加载
model, optimizer = accelerator.prepare(model, optimizer)
optimizer.load_state_dict(checkpoint["optimizer"])
```

---

## 11. 性能优化技巧

### 11.1 选择合适的 Micro Batch Size

TPU v5e 每个 core 约 16GB HBM。需要在内存允许的范围内选择最大的 micro batch size：

- 1 亿参数级别的模型：micro_batch_size = 32（seq_len=1024）比较安全
- 更大的模型需要调小，或开启梯度检查点
- 实际取值需要在 TPU 上试跑确认不 OOM

### 11.2 bf16 vs fp32

```python
# bf16 训练吞吐量约为 fp32 的 1.5-2x
accelerator = Accelerator(mixed_precision="bf16")
```

### 11.3 减少 all-reduce 次数

使用 `accelerator.accumulate(model)` 上下文管理器，只在累积的最后一步做梯度同步。

### 11.4 DataLoader 的 num_workers

在 TPU 上，数据加载通常不是瓶颈（TPU 计算比 GPU 慢，数据准备跟得上）。如果数据加载成为瓶颈，可以用多 worker 的 DataLoader，但要注意：

```python
# Kaggle TPU 环境下，简单的方式往往最稳定
# 直接在主进程中加载 numpy shard 文件通常够用
arr = np.load(filename).astype(np.int32)
tokens = torch.tensor(arr, dtype=torch.long)
```

### 11.5 监控吞吐量

```python
t0 = time.time()
# ... 训练一步 ...
t1 = time.time()

tokens_processed = MICRO_BATCH_SIZE * SEQ_LEN * grad_accum_steps * world_size
tok_per_sec = tokens_processed / (t1 - t0)
print(f"tok/s: {tok_per_sec:.0f}")
```

在 TPU v5e-8 上，1 亿参数级别的语言模型可以达到约 40 万 tokens/s 的训练吞吐量。

---

## 12. 完整最小示例

一个可以在 Kaggle TPU 上运行的最小训练循环：

```python
# Cell 1: 环境准备
import os
os.environ["PJRT_DEVICE"] = "TPU"
os.environ["OMP_NUM_THREADS"] = "1"
if "TPU_PROCESS_ADDRESSES" in os.environ:
    del os.environ["TPU_PROCESS_ADDRESSES"]
if "CLOUD_TPU_TASK_ID" in os.environ:
    del os.environ["CLOUD_TPU_TASK_ID"]
os.environ['XLA_TENSOR_ALLOCATOR_MAXSIZE'] = '100000000'

# Cell 2: 写入训练脚本
%%writefile train_minimal.py
import os
os.environ['XLA_TENSOR_ALLOCATOR_MAXSIZE'] = '100000000'

import torch
import torch.nn as nn
from torch.nn import functional as F
from accelerate import Accelerator, notebook_launcher
from accelerate.utils import set_seed

def train_worker():
    # 1. 创建 Accelerator
    accelerator = Accelerator(
        mixed_precision="bf16",
        gradient_accumulation_steps=1,
    )
    use_xla_sync = accelerator.device.type == "xla"

    # 2. 设置种子
    set_seed(1337 + accelerator.process_index)
    device = accelerator.device

    # 3. 创建模型（简单示例）
    model = nn.Sequential(
        nn.Linear(100, 256),
        nn.ReLU(),
        nn.Linear(256, 10),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # 4. prepare
    model, optimizer = accelerator.prepare(model, optimizer)

    if use_xla_sync:
        import torch_xla.core.xla_model as xm
        xm.mark_step()

    # 5. 训练循环
    for step in range(100):
        # 合成数据
        x = torch.randn(32, 100, device=device)
        y = torch.randint(0, 10, (32,), device=device)

        with accelerator.autocast():
            logits = model(x)
            loss = F.cross_entropy(logits, y)

        accelerator.backward(loss)
        optimizer.step()
        optimizer.zero_grad()

        if use_xla_sync:
            torch_xla.sync()

        if step % 10 == 0 and accelerator.process_index == 0:
            global_loss = accelerator.gather(loss.detach()).mean().item()
            print(f"step {step} | loss {global_loss:.4f}")

    if accelerator.process_index == 0:
        print("training done")

def launch():
    notebook_launcher(train_worker, num_processes=8)

if __name__ == "__main__":
    launch()

# Cell 3: 启动
from train_minimal import launch
launch()
```

---

## 13. 常见报错与解决方案

### 13.1 `RuntimeError: PJRT runtime not found`

**原因**：没有设置 `PJRT_DEVICE=TPU`。

**解决**：在所有 `import torch` 之前加上 `os.environ["PJRT_DEVICE"] = "TPU"`。

### 13.2 `RuntimeError: Bad StatusOr access` 或 TPU 初始化失败

**原因**：Kaggle 环境中的 `TPU_PROCESS_ADDRESSES` 或 `CLOUD_TPU_TASK_ID` 干扰了 XLA 的进程发现。

**解决**：在环境补丁中删除这两个变量。

### 13.3 NaN loss

**原因**：bf16 精度不够，LayerNorm / softmax / cross_entropy 中出现数值溢出。

**解决**：
1. 首先检查是否设置了 `XLA_USE_BF16=1`，有则立即删除（见 [4.4 节](#44-永远不要用-xla_use_bf16-全局开关实战教训)，它会使所有 fp32 保护失效）
2. 参考[第 4.2 节](#42-必须强制-fp32-的关键路径)，在关键路径强制 fp32

### 13.4 OOM（Out of Memory）

**原因**：中间张量太大，或计算图累积未释放。

**解决**：
1. 调小 `MICRO_BATCH_SIZE`
2. 开启梯度检查点
3. 验证循环中每步 `torch_xla.sync()` 释放中间张量

### 13.5 训练死锁（所有进程卡住不动）

**原因**：某个分支只在部分进程执行（如 `if rank == 0:` 做推理），XLA all-reduce 等待所有进程参与。

**解决**：确保所有 TPU core 执行相同的计算图路径。只在打印/保存时区分 rank。

### 13.6 `accelerator.prepare()` 后权重值变了

**原因**：weight tying 被破坏，或 optimizer state 被重置。

**解决**：
1. 如果模型有 weight tying，改用 `F.linear(x, embedding.weight)` 代替独立的输出层
2. optimizer state 在 `prepare()` 之后加载

### 13.7 resume 后 loss 突然跳变

**原因**：optimizer state 没有正确加载，或数据位置不对。

**解决**：
1. 确认 optimizer state 在 `prepare()` 之后加载
2. 确认数据加载器的 `state_dict` 被正确保存和恢复
3. 加载后打印第一个参数的前几个值验证

### 13.8 `scaled_dot_product_attention` 报错或很慢

**原因**：SDPA 是 CUDA 优化的算子，在 XLA 上不支持或走 fallback。

**解决**：检测 `x.device.type != "xla"` 时才用 SDPA，否则用手写 attention。

