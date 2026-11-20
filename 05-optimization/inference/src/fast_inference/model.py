"""GPT matching 01-pretraining checkpoints, with KV caches for fast decoding."""
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    vocab_size: int = 50304
    sequence_length: int = 1024
    n_layer: int = 12
    n_head: int = 12
    n_kv_head: int = 0
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = False
    norm: str = "layernorm"
    norm_position: str = "pre"
    qk_norm: bool = False
    activation: str = "gelu"
    pos_emb: str = "learned"
    rope_theta: float = 10000.0
    tie_embeddings: bool = True
    logit_softcap: float = 0.0
    depth_scaled_init: bool = False


def make_norm(config, dim):
    return nn.RMSNorm(dim) if config.norm == "rmsnorm" else nn.LayerNorm(dim)


def precompute_rope(max_len, head_dim, theta):
    half = head_dim // 2
    inv_freq = 1.0 / (theta ** (torch.arange(half, dtype=torch.float32) / half))
    freqs = torch.outer(torch.arange(max_len, dtype=torch.float32), inv_freq)
    return freqs.cos(), freqs.sin()  # (max_len, D/2)


def apply_rope(x, cos, sin):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1).type_as(x)


class DynamicCache:
    """Grows by concatenation every step: simple, but reallocates and changes shapes."""

    def __init__(self):
        self.keys, self.values = None, None

    def update(self, input_pos, keys, values):
        if self.keys is not None:
            keys, values = torch.cat([self.keys, keys], dim=2), torch.cat([self.values, values], dim=2)
        self.keys, self.values = keys, values
        return keys, values


class StaticCache(nn.Module):
    """Preallocated to max_len: fixed shapes, so the decode step can be compiled and graph-captured."""

    def __init__(self, batch_size, n_kv_head, max_len, head_dim, dtype, device):
        super().__init__()
        shape = (batch_size, n_kv_head, max_len, head_dim)
        self.register_buffer("keys", torch.zeros(shape, dtype=dtype, device=device), persistent=False)
        self.register_buffer("values", torch.zeros(shape, dtype=dtype, device=device), persistent=False)
        self.max_len = max_len

    def update(self, input_pos, keys, values):
        self.keys.index_copy_(2, input_pos, keys)
        self.values.index_copy_(2, input_pos, values)
        return self.keys, self.values


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head or config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.qk_norm = config.qk_norm
        self.q_proj = nn.Linear(config.n_embd, self.n_head * self.head_dim, bias=config.bias)
        self.k_proj = nn.Linear(config.n_embd, self.n_kv_head * self.head_dim, bias=config.bias)
        self.v_proj = nn.Linear(config.n_embd, self.n_kv_head * self.head_dim, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.cache = None

    def forward(self, x, input_pos, rope):
        batch, seq, _ = x.shape
        q = self.q_proj(x).view(batch, seq, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, seq, self.n_kv_head, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, seq, self.n_kv_head, self.head_dim).transpose(1, 2)
        if self.qk_norm:
            q, k = F.rms_norm(q, (self.head_dim,)), F.rms_norm(k, (self.head_dim,))
        if rope is not None:
            q, k = apply_rope(q, *rope), apply_rope(k, *rope)

        gqa = self.n_kv_head != self.n_head
        if isinstance(self.cache, StaticCache):
            k, v = self.cache.update(input_pos, k, v)
            mask = torch.arange(self.cache.max_len, device=x.device)[None, :] <= input_pos[:, None]  # (T, max_len)
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, enable_gqa=gqa)
        else:
            if self.cache is not None:
                k, v = self.cache.update(input_pos, k, v)
            y = F.scaled_dot_product_attention(q, k, v, is_causal=seq > 1, enable_gqa=gqa)  # decode: attend to all
        return self.c_proj(y.transpose(1, 2).contiguous().view(batch, seq, -1))


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.activation = config.activation
        self.gated = config.activation in ("swiglu", "geglu")
        hidden = 64 * math.ceil(8 * config.n_embd / 3 / 64) if self.gated else 4 * config.n_embd
        self.c_fc = nn.Linear(config.n_embd, 2 * hidden if self.gated else hidden, bias=config.bias)
        self.c_proj = nn.Linear(hidden, config.n_embd, bias=config.bias)

    def forward(self, x):
        hidden = self.c_fc(x)
        if self.gated:
            gate, value = hidden.chunk(2, dim=-1)
            hidden = (F.silu(gate) if self.activation == "swiglu" else F.gelu(gate)) * value
        elif self.activation == "relu2":
            hidden = F.relu(hidden).square()
        else:
            hidden = F.gelu(hidden)
        return self.c_proj(hidden)


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.post_norm = config.norm_position == "post"
        self.ln_1 = make_norm(config, config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = make_norm(config, config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x, input_pos, rope):
        if self.post_norm:
            x = self.ln_1(x + self.attn(x, input_pos, rope))
            return self.ln_2(x + self.mlp(x))
        x = x + self.attn(self.ln_1(x), input_pos, rope)
        return x + self.mlp(self.ln_2(x))


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.sequence_length, config.n_embd) if config.pos_emb == "learned" else None
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = make_norm(config, config.n_embd) if config.norm_position == "pre" else nn.Identity()
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        if config.tie_embeddings:
            self.wte.weight = self.lm_head.weight
        if config.pos_emb == "rope":
            cos, sin = precompute_rope(config.sequence_length, config.n_embd // config.n_head, config.rope_theta)
            self.register_buffer("rope_cos", cos, persistent=False)
            self.register_buffer("rope_sin", sin, persistent=False)

    def setup_cache(self, mode, batch_size=1, max_len=None):
        """mode: none (recompute everything), dynamic (concat cache), static (preallocated cache)."""
        param = self.lm_head.weight
        for block in self.blocks:
            attention = block.attn
            if mode == "static":
                attention.cache = StaticCache(batch_size, attention.n_kv_head, max_len or self.config.sequence_length,
                                              attention.head_dim, param.dtype, param.device)
            else:
                attention.cache = DynamicCache() if mode == "dynamic" else None

    def forward(self, idx, input_pos, last_only=False):
        x = self.wte(idx)
        if self.wpe is not None:
            x = x + self.wpe(input_pos)
        rope = (self.rope_cos[input_pos], self.rope_sin[input_pos]) if self.config.pos_emb == "rope" else None
        for block in self.blocks:
            x = block(x, input_pos, rope)
        if last_only:
            x = x[:, -1:]  # generation needs the next token only: skip the vocab projection elsewhere
        logits = self.lm_head(self.ln_f(x))
        if self.config.logit_softcap > 0:
            logits = self.config.logit_softcap * torch.tanh(logits / self.config.logit_softcap)
        return logits


def load_model(ckpt_path, device, dtype=torch.bfloat16):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model = GPT(GPTConfig(**ckpt["config"]))
    model.load_state_dict(ckpt["model"])
    return model.to(device=device, dtype=dtype).eval(), ckpt.get("tokenizer", "gpt2")