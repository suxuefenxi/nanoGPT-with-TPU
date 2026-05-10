import os
import math
import time
import inspect
from dataclasses import dataclass
from contextlib import nullcontext

import numpy as np
import tiktoken
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group


# =========================
# Global Config
# =========================

DATA_ROOT = "edu_fineweb10B"
LOG_DIR = "log"
LOG_FILE = os.path.join(LOG_DIR, "log.txt")

SEED = 1337

TOTAL_BATCH_SIZE = 524288
MICRO_BATCH_SIZE = 64
SEQ_LEN = 1024

MODEL_VOCAB_SIZE = 50304

WEIGHT_DECAY = 0.1
GRAD_CLIP = 1.0

MAX_LR = 6e-4
MIN_LR = MAX_LR * 0.1
WARMUP_STEPS = 715
MAX_STEPS = 19073

VAL_INTERVAL = 250
VAL_STEPS = 20
CKPT_INTERVAL = 5000

USE_COMPILE = False

SAMPLE_INTERVAL = 250
SAMPLE_NUM_RETURN_SEQS = 4
SAMPLE_MAX_LENGTH = 32
SAMPLE_TOPK = 50
SAMPLE_PROMPT = "Hello, I'm a language model,"


# =========================
# Model
# =========================

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x):
        B, T, C = x.size()

        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)

        head_dim = C // self.n_head
        q = q.view(B, T, self.n_head, head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, head_dim).transpose(1, 2)

        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        return self.c_proj(y)


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate="tanh")
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


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


class GPT(nn.Module):
    def __init__(self, config):
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
        assert T <= self.config.block_size, (
            f"Cannot forward sequence of length {T}, "
            f"block size is only {self.config.block_size}"
        )

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

    @classmethod
    def from_pretrained(cls, model_type):
        assert model_type in {"gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"}
        from transformers import GPT2LMHeadModel

        print(f"loading weights from pretrained gpt: {model_type}")

        config_args = {
            "gpt2":        dict(n_layer=12, n_head=12, n_embd=768),
            "gpt2-medium": dict(n_layer=24, n_head=16, n_embd=1024),
            "gpt2-large":  dict(n_layer=36, n_head=20, n_embd=1280),
            "gpt2-xl":     dict(n_layer=48, n_head=25, n_embd=1600),
        }[model_type]

        config_args["vocab_size"] = 50257
        config_args["block_size"] = 1024

        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()

        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        sd_keys = [k for k in sd.keys() if not k.endswith(".attn.bias")]
        sd_keys_hf = [k for k in sd_hf.keys() if not k.endswith(".attn.masked_bias")]
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith(".attn.bias")]

        transposed = [
            "attn.c_attn.weight",
            "attn.c_proj.weight",
            "mlp.c_fc.weight",
            "mlp.c_proj.weight",
        ]

        assert len(sd_keys_hf) == len(sd_keys), (
            f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        )

        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    def configure_optimizers(self, weight_decay, learning_rate, device_type, master_process=True):
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
            print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
            print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")

        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"

        if master_process:
            print(f"using fused AdamW: {use_fused}")

        optimizer = torch.optim.AdamW(
            optim_groups,
            lr=learning_rate,
            betas=(0.9, 0.95),
            eps=1e-8,
            fused=use_fused,
        )
        return optimizer


# =========================
# Data
# =========================

def load_tokens(filename):
    npt = np.load(filename).astype(np.int32)
    return torch.tensor(npt, dtype=torch.long)


class DataLoaderLite:
    def __init__(self, B, T, process_rank, num_processes, split, data_root=DATA_ROOT, master_process=True):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        self.data_root = data_root

        assert split in {"train", "val"}
        self.split = split

        shards = os.listdir(data_root)
        shards = [s for s in shards if split in s]
        shards = sorted(shards)
        shards = [os.path.join(data_root, s) for s in shards]

        assert len(shards) > 0, f"no shards found for split {split}"
        self.shards = shards

        if master_process:
            print(f"found {len(shards)} shards for split {split}")

        self.reset()

    def reset(self):
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = self.B * self.T * self.process_rank

    def next_batch(self):
        B, T = self.B, self.T

        buf = self.tokens[self.current_position:self.current_position + B * T + 1]
        x = buf[:-1].view(B, T)
        y = buf[1:].view(B, T)

        self.current_position += B * T * self.num_processes

        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_position = B * T * self.process_rank

        return x, y


# =========================
# Utils
# =========================

def setup_distributed():
    ddp = int(os.environ.get("RANK", -1)) != -1

    if ddp:
        assert torch.cuda.is_available(), "DDP currently requires CUDA"
        init_process_group(backend="nccl")

        ddp_rank = int(os.environ["RANK"])
        ddp_local_rank = int(os.environ["LOCAL_RANK"])
        ddp_world_size = int(os.environ["WORLD_SIZE"])

        device = f"cuda:{ddp_local_rank}"
        torch.cuda.set_device(device)
        master_process = (ddp_rank == 0)
    else:
        ddp_rank = 0
        ddp_local_rank = 0
        ddp_world_size = 1
        master_process = True

        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

        print(f"using device: {device}")

    if str(device).startswith("cuda"):
        device_type = "cuda"
    elif str(device) == "mps":
        device_type = "mps"
    else:
        device_type = "cpu"

    return ddp, ddp_rank, ddp_local_rank, ddp_world_size, master_process, device, device_type


def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def get_lr(step):
    if step < WARMUP_STEPS:
        return MAX_LR * (step + 1) / WARMUP_STEPS

    if step > MAX_STEPS:
        return MIN_LR

    decay_ratio = (step - WARMUP_STEPS) / (MAX_STEPS - WARMUP_STEPS)
    assert 0 <= decay_ratio <= 1

    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return MIN_LR + coeff * (MAX_LR - MIN_LR)


def get_autocast_context(device_type):
    if device_type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def append_log(log_file, text, master_process):
    if not master_process:
        return
    with open(log_file, "a") as f:
        f.write(text + "\n")


def evaluate(model, val_loader, device, device_type, ddp):
    model.eval()
    val_loader.reset()

    val_loss_accum = torch.zeros((), device=device)

    with torch.no_grad():
        for _ in range(VAL_STEPS):
            x, y = val_loader.next_batch()
            x, y = x.to(device), y.to(device)

            with get_autocast_context(device_type):
                _, loss = model(x, y)

            val_loss_accum += loss.detach() / VAL_STEPS

    if ddp:
        dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)

    return val_loss_accum.item()


def generate_samples(model, enc, device, device_type, rank=0):
    model.eval()

    torch.manual_seed(42 + rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42 + rank)

    tokens = enc.encode(SAMPLE_PROMPT)
    tokens = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
    xgen = tokens.repeat(SAMPLE_NUM_RETURN_SEQS, 1)

    while xgen.size(1) < SAMPLE_MAX_LENGTH:
        with torch.no_grad():
            with get_autocast_context(device_type):
                logits, _ = model(xgen)

            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            topk_probs, topk_indices = torch.topk(probs, SAMPLE_TOPK, dim=-1)
            ix = torch.multinomial(topk_probs, 1)
            xcol = torch.gather(topk_indices, -1, ix)
            xgen = torch.cat((xgen, xcol), dim=1)

    for i in range(SAMPLE_NUM_RETURN_SEQS):
        tokens = xgen[i, :SAMPLE_MAX_LENGTH].tolist()
        decoded = enc.decode(tokens)
        print(f"sample {i}: {decoded}")


def save_checkpoint(raw_model, step, val_loss):
    checkpoint_path = os.path.join(LOG_DIR, f"model_{step:05d}.pt")
    checkpoint = {
        "model": raw_model.state_dict(),
        "config": raw_model.config,
        "step": step,
        "val_loss": val_loss,
    }
    torch.save(checkpoint, checkpoint_path)


# =========================
# Train
# =========================

def main():
    ddp, ddp_rank, ddp_local_rank, ddp_world_size, master_process, device, device_type = setup_distributed()

    try:
        set_seed(SEED)
        torch.set_float32_matmul_precision("high")

        enc = tiktoken.get_encoding("gpt2")

        assert TOTAL_BATCH_SIZE % (MICRO_BATCH_SIZE * SEQ_LEN * ddp_world_size) == 0, (
            "TOTAL_BATCH_SIZE must be divisible by MICRO_BATCH_SIZE * SEQ_LEN * ddp_world_size"
        )
        grad_accum_steps = TOTAL_BATCH_SIZE // (MICRO_BATCH_SIZE * SEQ_LEN * ddp_world_size)

        if master_process:
            print(f"total desired batch size: {TOTAL_BATCH_SIZE}")
            print(f"=> calculated gradient accumulation steps: {grad_accum_steps}")

        train_loader = DataLoaderLite(
            B=MICRO_BATCH_SIZE,
            T=SEQ_LEN,
            process_rank=ddp_rank,
            num_processes=ddp_world_size,
            split="train",
            master_process=master_process,
        )
        val_loader = DataLoaderLite(
            B=MICRO_BATCH_SIZE,
            T=SEQ_LEN,
            process_rank=ddp_rank,
            num_processes=ddp_world_size,
            split="val",
            master_process=master_process,
        )

        model = GPT(GPTConfig(vocab_size=MODEL_VOCAB_SIZE))
        model.to(device)

        if USE_COMPILE:
            model = torch.compile(model)

        if ddp:
            model = DDP(model, device_ids=[ddp_local_rank])

        raw_model = model.module if ddp else model

        optimizer = raw_model.configure_optimizers(
            weight_decay=WEIGHT_DECAY,
            learning_rate=MAX_LR,
            device_type=device_type,
            master_process=master_process,
        )

        os.makedirs(LOG_DIR, exist_ok=True)
        if master_process:
            with open(LOG_FILE, "w") as f:
                pass

        for step in range(MAX_STEPS):
            t0 = time.time()
            last_step = (step == MAX_STEPS - 1)

            if step % VAL_INTERVAL == 0 or last_step:
                val_loss = evaluate(model, val_loader, device, device_type, ddp)

                if master_process:
                    print(f"validation loss: {val_loss:.4f}")
                    append_log(LOG_FILE, f"{step} val {val_loss:.4f}", master_process)

                    if step > 0 and (step % CKPT_INTERVAL == 0 or last_step):
                        save_checkpoint(raw_model, step, val_loss)

            if ((step > 0 and step % SAMPLE_INTERVAL == 0) or last_step) and (not USE_COMPILE):
                if master_process:
                    generate_samples(raw_model, enc, device, device_type, rank=ddp_rank)

            model.train()
            optimizer.zero_grad(set_to_none=True)
            loss_accum = torch.zeros((), device=device)

            for micro_step in range(grad_accum_steps):
                x, y = train_loader.next_batch()
                x, y = x.to(device), y.to(device)

                if ddp:
                    model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)

                with get_autocast_context(device_type):
                    _, loss = model(x, y)

                loss = loss / grad_accum_steps
                loss_accum += loss.detach()
                loss.backward()

            if ddp:
                dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)

            norm = torch.nn.utils.clip_grad_norm_(raw_model.parameters(), GRAD_CLIP)

            lr = get_lr(step)
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

            optimizer.step()

            if device_type == "cuda":
                torch.cuda.synchronize()

            t1 = time.time()
            dt = t1 - t0
            tokens_processed = MICRO_BATCH_SIZE * SEQ_LEN * grad_accum_steps * ddp_world_size
            tokens_per_sec = tokens_processed / dt

            if master_process:
                print(
                    f"step {step:5d} | "
                    f"loss: {loss_accum.item():.6f} | "
                    f"lr {lr:.4e} | "
                    f"norm: {float(norm):.4f} | "
                    f"dt: {dt * 1000:.2f}ms | "
                    f"tok/sec: {tokens_per_sec:.2f}"
                )
                append_log(LOG_FILE, f"{step} train {loss_accum.item():.6f}", master_process)

    finally:
        if ddp:
            destroy_process_group()


if __name__ == "__main__":
    main()