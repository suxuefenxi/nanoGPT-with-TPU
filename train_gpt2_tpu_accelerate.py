import os
import math
import time
import inspect
from dataclasses import dataclass, asdict
from contextlib import nullcontext

import numpy as np
import tiktoken
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn import functional as F

from accelerate import Accelerator, notebook_launcher
from accelerate.utils import set_seed


# ============================================================
# Global Config
# ============================================================

DATA_ROOT = "/kaggle/input/datasets/zhxlidf/edu-fineweb10b"  
OUT_DIR = "log_tpu"
BEST_CKPT_NAME = "best_ckpt.pt"
FINAL_CKPT_NAME = "final_ckpt.pt"

SEED = 1337

# 训练 batch 配置
TOTAL_BATCH_SIZE = 524288   # 全局 token batch
MICRO_BATCH_SIZE = 64       # 每个进程每次 forward 的 batch size，根据显存大小调整
SEQ_LEN = 1024

# 模型配置
MODEL_VOCAB_SIZE = 50304
N_LAYER = 12
N_HEAD = 12
N_EMBD = 768

# 优化器
WEIGHT_DECAY = 0.1
BETA1 = 0.9
BETA2 = 0.95
EPS = 1e-8
GRAD_CLIP = 1.0

# 学习率
MAX_LR = 6e-4
MIN_LR = MAX_LR * 0.1
WARMUP_STEPS = 715
MAX_STEPS = 19073

# 验证 / 保存
VAL_INTERVAL = 500
VAL_STEPS = 20
SAVE_BEST_ONLY_WHEN_IMPROVED = True

# 采样
DO_SAMPLE = True
SAMPLE_INTERVAL = 500
SAMPLE_NUM_RETURN_SEQS = 4
SAMPLE_MAX_LENGTH = 32
SAMPLE_TOPK = 50
SAMPLE_PROMPT = "Hello, I'm a language model,"

# TPU / Accelerate
TPU_NUM_PROCESSES = 8
MIXED_PRECISION = "bf16"   # "bf16" or "no"
USE_TORCH_COMPILE = False  # TPU 不建议开

# Attention
# TPU 上默认走手写 causal attention，不走 flash / fused / cuda sdpa 快路径
USE_SDPA = False

# wandb
WANDB_ENABLED = True
WANDB_PROJECT = "nanogpt-tpu"
WANDB_RUN_NAME = "nanogpt-kaggle-tpuv5e-8"
SWANLAB_API_KEY = os.environ.get("SWANLAB_API_KEY", None)

# resume
RESUME_PATH = None  # 例如 "log_tpu/best_ckpt.pt"


# ============================================================
# Utils
# ============================================================

def get_lr(step: int) -> float:
    if step < WARMUP_STEPS:
        return MAX_LR * (step + 1) / WARMUP_STEPS
    if step > MAX_STEPS:
        return MIN_LR

    decay_ratio = (step - WARMUP_STEPS) / (MAX_STEPS - WARMUP_STEPS)
    decay_ratio = min(max(decay_ratio, 0.0), 1.0)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return MIN_LR + coeff * (MAX_LR - MIN_LR)


def tree_to_cpu(obj):
    if torch.is_tensor(obj):
        return obj.detach().cpu()
    if isinstance(obj, dict):
        return {k: tree_to_cpu(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [tree_to_cpu(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(tree_to_cpu(v) for v in obj)
    return obj


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def load_tokens(filename):
    arr = np.load(filename).astype(np.int32)
    return torch.tensor(arr, dtype=torch.long)


# ============================================================
# Data Loader
# ============================================================

class StatefulShardLoader:
    """
    顺序读取 shard，并按多进程切分 batch。
    这里保存的是“全局同步状态”：
      - current_shard
      - batch_idx_in_shard
    这样 rank0 保存的 checkpoint 可以让所有 rank 正确 resume。
    """

    def __init__(self, B, T, process_rank, num_processes, split, data_root=DATA_ROOT, master_process=True):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        self.split = split
        self.data_root = data_root

        assert split in {"train", "val"}

        shards = os.listdir(data_root)
        shards = [s for s in shards if split in s]
        shards = sorted(shards)
        shards = [os.path.join(data_root, s) for s in shards]
        assert len(shards) > 0, f"no shards found for split={split} in {data_root}"

        self.shards = shards
        self.global_stride = self.B * self.T * self.num_processes

        if master_process:
            print(f"[{split}] found {len(shards)} shards", flush=True)

        self.reset()

    def reset(self):
        self.current_shard = 0
        self.batch_idx_in_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])

    def _advance_shard(self):
        self.current_shard = (self.current_shard + 1) % len(self.shards)
        self.batch_idx_in_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])

    def _has_enough_tokens_for_synced_batch(self):
        # 第 n 个同步 batch 需要的最大 token 索引上界：
        # (n+1) * global_stride + 1
        needed = (self.batch_idx_in_shard + 1) * self.global_stride + 1
        return needed <= len(self.tokens)

    def next_batch(self):
        while not self._has_enough_tokens_for_synced_batch():
            self._advance_shard()

        start = self.batch_idx_in_shard * self.global_stride + self.process_rank * self.B * self.T
        end = start + self.B * self.T + 1

        buf = self.tokens[start:end]
        x = buf[:-1].view(self.B, self.T)
        y = buf[1:].view(self.B, self.T)

        self.batch_idx_in_shard += 1
        return x, y

    def state_dict(self):
        return {
            "current_shard": self.current_shard,
            "batch_idx_in_shard": self.batch_idx_in_shard,
        }

    def load_state_dict(self, state):
        self.current_shard = int(state["current_shard"])
        self.batch_idx_in_shard = int(state["batch_idx_in_shard"])
        self.tokens = load_tokens(self.shards[self.current_shard])


# ============================================================
# Model
# ============================================================

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.use_sdpa = config.use_sdpa

        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.block_size, config.block_size))
            .view(1, 1, config.block_size, config.block_size),
            persistent=False,
        )

    def forward(self, x):
        B, T, C = x.size()

        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)

        head_dim = C // self.n_head
        q = q.view(B, T, self.n_head, head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, head_dim).transpose(1, 2)

        if self.use_sdpa and x.device.type != "xla":
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
            att = F.softmax(att, dim=-1)
            y = att @ v

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate="tanh")
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        return self.c_proj(self.gelu(self.c_fc(x)))


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    use_sdpa: bool = False


class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(config.vocab_size, config.n_embd),
            "wpe": nn.Embedding(config.block_size, config.n_embd),
            "h": nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            "ln_f": nn.LayerNorm(config.n_embd),
        })
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.transformer.wte.weight = self.lm_head.weight
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, "NANOGPT_SCALE_INIT"):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.size()
        assert T <= self.config.block_size, f"T={T} > block_size={self.config.block_size}"

        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = tok_emb + pos_emb

        for block in self.transformer.h:
            x = block(x)

        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1)
            )
        return logits, loss

    def configure_optimizers(self, weight_decay, learning_rate, betas=(0.9, 0.95), eps=1e-8, master_process=True):
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}

        decay_params = [p for _, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for _, p in param_dict.items() if p.dim() < 2]

        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]

        if master_process:
            num_decay_params = sum(p.numel() for p in decay_params)
            num_nodecay_params = sum(p.numel() for p in nodecay_params)
            print(f"num decayed tensors: {len(decay_params)} | params: {num_decay_params:,}", flush=True)
            print(f"num non-decayed tensors: {len(nodecay_params)} | params: {num_nodecay_params:,}", flush=True)

        # TPU / XLA 不用 fused AdamW
        optimizer = torch.optim.AdamW(
            optim_groups,
            lr=learning_rate,
            betas=betas,
            eps=eps,
        )
        return optimizer


# ============================================================
# Eval / Sample / Checkpoint
# ============================================================

@torch.no_grad()
def evaluate(model, val_loader, accelerator, val_steps, use_xla_sync=False):
    model.eval()
    val_loader.reset()

    losses = []

    for _ in range(val_steps):
        x, y = val_loader.next_batch()
        x = x.to(accelerator.device)
        y = y.to(accelerator.device)

        with accelerator.autocast():
            _, loss = model(x, y)

        losses.append(loss.detach())

        if use_xla_sync:
            import torch_xla
            torch_xla.sync()

    local_mean = torch.stack(losses).mean()
    global_mean = accelerator.gather(local_mean).mean().item()

    model.train()
    return global_mean


@torch.no_grad()
def generate_samples(model, enc, accelerator, step):
    model.eval()

    tokens = enc.encode(SAMPLE_PROMPT)
    xgen = torch.tensor(tokens, dtype=torch.long, device=accelerator.device).unsqueeze(0)
    xgen = xgen.repeat(SAMPLE_NUM_RETURN_SEQS, 1)

    sample_gen = torch.Generator(device=accelerator.device)
    sample_gen.manual_seed(42)

    while xgen.size(1) < SAMPLE_MAX_LENGTH:
        with accelerator.autocast():
            logits, _ = model(xgen)

        logits = logits[:, -1, :]
        probs = F.softmax(logits, dim=-1)
        topk_probs, topk_indices = torch.topk(probs, SAMPLE_TOPK, dim=-1)
        ix = torch.multinomial(topk_probs, 1, generator=sample_gen)
        xcol = torch.gather(topk_indices, -1, ix)
        xgen = torch.cat((xgen, xcol), dim=1)

    if accelerator.process_index == 0:
        print(f"\n===== sample @ step {step} =====", flush=True)
        for i in range(SAMPLE_NUM_RETURN_SEQS):
            out = enc.decode(xgen[i].tolist())
            print(f"sample {i}: {out}", flush=True)
        print("================================\n", flush=True)

    model.train()


def save_checkpoint(
    accelerator,
    model,
    optimizer,
    train_loader,
    val_loader,
    step,
    best_val_loss,
    model_config,
    save_path,
):
    accelerator.wait_for_everyone()

    state_dict = accelerator.get_state_dict(model)
    opt_state = tree_to_cpu(optimizer.state_dict())

    ckpt = {
        "model": state_dict,
        "optimizer": opt_state,
        "step": step,
        "best_val_loss": best_val_loss,
        "model_config": asdict(model_config),
        "train_loader_state": train_loader.state_dict(),
        "val_loader_state": val_loader.state_dict(),
        "global_config": {
            "TOTAL_BATCH_SIZE": TOTAL_BATCH_SIZE,
            "MICRO_BATCH_SIZE": MICRO_BATCH_SIZE,
            "SEQ_LEN": SEQ_LEN,
            "MAX_LR": MAX_LR,
            "MIN_LR": MIN_LR,
            "WARMUP_STEPS": WARMUP_STEPS,
            "MAX_STEPS": MAX_STEPS,
            "WEIGHT_DECAY": WEIGHT_DECAY,
            "BETA1": BETA1,
            "BETA2": BETA2,
            "EPS": EPS,
            "GRAD_CLIP": GRAD_CLIP,
            "MIXED_PRECISION": MIXED_PRECISION,
        },
    }

    accelerator.save(ckpt, save_path)
    accelerator.wait_for_everyone()


# ============================================================
# Train
# ============================================================

def train_worker():
    import builtins

    ensure_dir(OUT_DIR)

    accelerator = Accelerator(
        mixed_precision=MIXED_PRECISION,
    )

    use_xla_sync = accelerator.device.type == "xla"

    def tpu_print(*args, **kwargs):
        if accelerator.process_index == 0:
            kwargs["flush"] = True
            builtins.print(*args, **kwargs)

    process_seed = SEED + accelerator.process_index
    set_seed(process_seed)

    world_size = accelerator.num_processes
    device = accelerator.device

    assert TOTAL_BATCH_SIZE % (MICRO_BATCH_SIZE * SEQ_LEN * world_size) == 0, (
        f"TOTAL_BATCH_SIZE={TOTAL_BATCH_SIZE} must be divisible by "
        f"MICRO_BATCH_SIZE*SEQ_LEN*world_size = {MICRO_BATCH_SIZE * SEQ_LEN * world_size}"
    )
    grad_accum_steps = TOTAL_BATCH_SIZE // (MICRO_BATCH_SIZE * SEQ_LEN * world_size)

    tpu_print(f"device = {device}")
    tpu_print(f"world_size = {world_size}")
    tpu_print(f"process_index = {accelerator.process_index}")
    tpu_print(f"mixed_precision = {MIXED_PRECISION}")
    tpu_print(f"grad_accum_steps(local) = {grad_accum_steps}")
    tpu_print(f"total batch size(tokens) = {TOTAL_BATCH_SIZE}")

    wandb = None
    if WANDB_ENABLED and accelerator.process_index == 0:
        try:
            import swanlab as wandb

            if SWANLAB_API_KEY:
                wandb.login(SWANLAB_API_KEY)

            wandb.init(
                project=WANDB_PROJECT,
                name=WANDB_RUN_NAME,
                config={
                    "data_root": DATA_ROOT,
                    "total_batch_size": TOTAL_BATCH_SIZE,
                    "micro_batch_size": MICRO_BATCH_SIZE,
                    "seq_len": SEQ_LEN,
                    "grad_accum_steps_local": grad_accum_steps,
                    "world_size": world_size,
                    "n_layer": N_LAYER,
                    "n_head": N_HEAD,
                    "n_embd": N_EMBD,
                    "vocab_size": MODEL_VOCAB_SIZE,
                    "weight_decay": WEIGHT_DECAY,
                    "beta1": BETA1,
                    "beta2": BETA2,
                    "eps": EPS,
                    "max_lr": MAX_LR,
                    "min_lr": MIN_LR,
                    "warmup_steps": WARMUP_STEPS,
                    "max_steps": MAX_STEPS,
                    "mixed_precision": MIXED_PRECISION,
                    "use_sdpa": USE_SDPA,
                    "resume_path": RESUME_PATH,
                },
            )
            tpu_print("wandb initialized")
        except Exception as e:
            tpu_print(f"wandb init failed, continue without wandb: {e}")
            wandb = None

    accelerator.wait_for_everyone()

    # -------------------------
    # Model / Optimizer / Resume
    # -------------------------

    start_step = 0
    best_val_loss = float("inf")
    checkpoint = None

    model_config = GPTConfig(
        block_size=SEQ_LEN,
        vocab_size=MODEL_VOCAB_SIZE,
        n_layer=N_LAYER,
        n_head=N_HEAD,
        n_embd=N_EMBD,
        use_sdpa=USE_SDPA,
    )

    if RESUME_PATH is not None:
        tpu_print(f"resuming from {RESUME_PATH}")
        checkpoint = torch.load(RESUME_PATH, map_location="cpu")

        if "model_config" in checkpoint:
            model_config = GPTConfig(**checkpoint["model_config"])

        model = GPT(model_config)
        model.load_state_dict(checkpoint["model"], strict=True)

        start_step = int(checkpoint["step"])
        best_val_loss = float(checkpoint["best_val_loss"])
    else:
        tpu_print("initializing model from scratch")
        model = GPT(model_config)

    if USE_TORCH_COMPILE:
        tpu_print("warning: torch.compile is disabled on TPU in practice; keeping it off is recommended")
        model = torch.compile(model)

    optimizer = model.configure_optimizers(
        weight_decay=WEIGHT_DECAY,
        learning_rate=MAX_LR,
        betas=(BETA1, BETA2),
        eps=EPS,
        master_process=(accelerator.process_index == 0),
    )

    model, optimizer = accelerator.prepare(model, optimizer)

    # optimizer 状态必须在 prepare 之后加载
    if checkpoint is not None and "optimizer" in checkpoint:
        tpu_print("loading optimizer state...")
        optimizer.load_state_dict(checkpoint["optimizer"])

    # -------------------------
    # Data
    # -------------------------

    train_loader = StatefulShardLoader(
        B=MICRO_BATCH_SIZE,
        T=SEQ_LEN,
        process_rank=accelerator.process_index,
        num_processes=world_size,
        split="train",
        data_root=DATA_ROOT,
        master_process=(accelerator.process_index == 0),
    )
    val_loader = StatefulShardLoader(
        B=MICRO_BATCH_SIZE,
        T=SEQ_LEN,
        process_rank=accelerator.process_index,
        num_processes=world_size,
        split="val",
        data_root=DATA_ROOT,
        master_process=(accelerator.process_index == 0),
    )

    if checkpoint is not None:
        if "train_loader_state" in checkpoint:
            train_loader.load_state_dict(checkpoint["train_loader_state"])
        if "val_loader_state" in checkpoint:
            val_loader.load_state_dict(checkpoint["val_loader_state"])

    enc = tiktoken.get_encoding("gpt2")

    # -------------------------
    # Train Loop
    # -------------------------

    raw_model = accelerator.unwrap_model(model)
    step = start_step

    tpu_print(f"start_step = {start_step}")
    tpu_print(f"best_val_loss = {best_val_loss}")

    while step < MAX_STEPS:
        t0 = time.time()
        last_step = (step == MAX_STEPS - 1)

        # eval
        if step % VAL_INTERVAL == 0 or last_step:
            val_loss = evaluate(
                model=model,
                val_loader=val_loader,
                accelerator=accelerator,
                val_steps=VAL_STEPS,
                use_xla_sync=use_xla_sync,
            )

            if accelerator.process_index == 0:
                tpu_print(f"step {step} | val_loss {val_loss:.4f}")
                if wandb is not None:
                    wandb.log({"val/loss": val_loss, "step": step}, step=step)

            improved = val_loss < best_val_loss
            if improved:
                best_val_loss = val_loss

            if step > 0 and improved and SAVE_BEST_ONLY_WHEN_IMPROVED:
                best_path = os.path.join(OUT_DIR, BEST_CKPT_NAME)
                tpu_print(f"saving best checkpoint to {best_path}")
                save_checkpoint(
                    accelerator=accelerator,
                    model=model,
                    optimizer=optimizer,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    step=step,
                    best_val_loss=best_val_loss,
                    model_config=raw_model.config,
                    save_path=best_path,
                )

        # sample
        if DO_SAMPLE and ((step > 0 and step % SAMPLE_INTERVAL == 0) or last_step):
            accelerator.wait_for_everyone()
            if accelerator.process_index == 0:
                generate_samples(raw_model, enc, accelerator, step)
            accelerator.wait_for_everyone()

        # train one optimizer step
        model.train()
        optimizer.zero_grad(set_to_none=True)

        loss_accum = torch.zeros((), device=device)

        lr = get_lr(step)
        for group in optimizer.param_groups:
            group["lr"] = lr

        for micro_step in range(grad_accum_steps):
            x, y = train_loader.next_batch()
            x = x.to(device)
            y = y.to(device)

            with accelerator.autocast():
                _, loss = model(x, y)

            loss_accum += loss.detach() / grad_accum_steps
            accelerator.backward(loss / grad_accum_steps)

        grad_norm = None
        if GRAD_CLIP is not None and GRAD_CLIP > 0:
            grad_norm = accelerator.clip_grad_norm_(model.parameters(), GRAD_CLIP)

        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        if use_xla_sync:
            import torch_xla
            torch_xla.sync()

        global_loss = accelerator.gather(loss_accum).mean().item()
        global_grad_norm = None
        if grad_norm is not None:
            grad_norm_tensor = grad_norm.detach() if torch.is_tensor(grad_norm) else torch.tensor(float(grad_norm), device=device)
            global_grad_norm = accelerator.gather(grad_norm_tensor).mean().item()

        t1 = time.time()
        dt = t1 - t0
        tokens_processed = MICRO_BATCH_SIZE * SEQ_LEN * grad_accum_steps * world_size
        tok_per_sec = tokens_processed / max(dt, 1e-6)

        if accelerator.process_index == 0:
            msg = (
                f"step {step:5d} | "
                f"loss {global_loss:.6f} | "
                f"lr {lr:.4e} | "
                f"grad_norm {global_grad_norm if global_grad_norm is not None else float('nan'):.4f} | "
                f"dt {dt * 1000:.2f}ms | "
                f"tok/s {tok_per_sec:.2f}"
            )
            tpu_print(msg)

            if wandb is not None:
                wandb.log(
                    {
                        "train/loss": global_loss,
                        "train/lr": lr,
                        "train/grad_norm": global_grad_norm if global_grad_norm is not None else float("nan"),
                        "perf/tok_per_sec": tok_per_sec,
                        "perf/step_time_ms": dt * 1000.0,
                        "step": step,
                    },
                    step=step,
                )

        step += 1

    # -------------------------
    # Final Checkpoint
    # -------------------------

    final_path = os.path.join(OUT_DIR, FINAL_CKPT_NAME)
    tpu_print(f"saving final checkpoint to {final_path}")
    save_checkpoint(
        accelerator=accelerator,
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        step=step,
        best_val_loss=best_val_loss,
        model_config=raw_model.config,
        save_path=final_path,
    )

    if accelerator.process_index == 0 and wandb is not None:
        wandb.finish()

    accelerator.wait_for_everyone()
    tpu_print("training done")


# ============================================================
# Launch
# ============================================================

def launch():
    notebook_launcher(train_worker, num_processes=TPU_NUM_PROCESSES)


if __name__ == "__main__":
    launch()