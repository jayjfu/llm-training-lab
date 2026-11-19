import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    vocab_size: int = 96
    sequence_length: int = 256
    n_layer: int = 4
    n_head: int = 4
    n_kv_head: int = 0  # 0 = same as n_head (MHA)
    n_embd: int = 128
    dropout: float = 0.0
    bias: bool = False
    norm: str = "layernorm"  # layernorm | rmsnorm
    norm_position: str = "pre"  # pre | post
    qk_norm: bool = False
    activation: str = "gelu"  # gelu | relu2 | swiglu | geglu
    pos_emb: str = "learned"  # learned | rope | nope
    rope_theta: float = 10000.0
    tie_embeddings: bool = True
    logit_softcap: float = 0.0
    depth_scaled_init: bool = False


def make_norm(config, dim):
    if config.norm == "rmsnorm":
        return nn.RMSNorm(dim)
    return nn.LayerNorm(dim)


def precompute_rope(sequence_length, head_dim, theta):
    half = head_dim // 2
    inv_freq = 1.0 / (theta ** (torch.arange(half, dtype=torch.float32) / half))
    freqs = torch.outer(torch.arange(sequence_length, dtype=torch.float32), inv_freq)
    return freqs.cos()[None, None], freqs.sin()[None, None]  # (1, 1, T, D/2)


def apply_rope(x, cos, sin):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1).type_as(x)


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head or config.n_head
        assert self.n_head % self.n_kv_head == 0
        self.head_dim = config.n_embd // config.n_head
        self.qk_norm = config.qk_norm
        self.dropout = config.dropout
        self.q_proj = nn.Linear(config.n_embd, self.n_head * self.head_dim, bias=config.bias)
        self.k_proj = nn.Linear(config.n_embd, self.n_kv_head * self.head_dim, bias=config.bias)
        self.v_proj = nn.Linear(config.n_embd, self.n_kv_head * self.head_dim, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(self, x, rope=None):
        batch, seq, _ = x.shape
        q = self.q_proj(x).view(batch, seq, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, seq, self.n_kv_head, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, seq, self.n_kv_head, self.head_dim).transpose(1, 2)
        if self.qk_norm:
            q, k = F.rms_norm(q, (self.head_dim,)), F.rms_norm(k, (self.head_dim,))
        if rope is not None:
            q, k = apply_rope(q, *rope), apply_rope(k, *rope)
        y = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
            enable_gqa=self.n_kv_head != self.n_head,
        )
        y = y.transpose(1, 2).contiguous().view(batch, seq, -1)
        return self.resid_dropout(self.c_proj(y))


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.activation = config.activation
        self.gated = config.activation in ("swiglu", "geglu")
        hidden = 64 * math.ceil(8 * config.n_embd / 3 / 64) if self.gated else 4 * config.n_embd  # param-matched
        self.c_fc = nn.Linear(config.n_embd, 2 * hidden if self.gated else hidden, bias=config.bias)
        self.c_proj = nn.Linear(hidden, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        hidden = self.c_fc(x)
        if self.gated:
            gate, value = hidden.chunk(2, dim=-1)
            hidden = (F.silu(gate) if self.activation == "swiglu" else F.gelu(gate)) * value
        elif self.activation == "relu2":
            hidden = F.relu(hidden).square()
        else:
            hidden = F.gelu(hidden)
        return self.dropout(self.c_proj(hidden))


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.post_norm = config.norm_position == "post"
        self.ln_1 = make_norm(config, config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = make_norm(config, config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x, rope=None):
        if self.post_norm:
            x = self.ln_1(x + self.attn(x, rope))
            return self.ln_2(x + self.mlp(x))
        x = x + self.attn(self.ln_1(x), rope)
        return x + self.mlp(self.ln_2(x))


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.sequence_length, config.n_embd) if config.pos_emb == "learned" else None
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = make_norm(config, config.n_embd) if config.norm_position == "pre" else nn.Identity()
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        if config.tie_embeddings:
            self.wte.weight = self.lm_head.weight
        if config.pos_emb == "rope":
            cos, sin = precompute_rope(config.sequence_length, config.n_embd // config.n_head, config.rope_theta)
            self.register_buffer("rope_cos", cos, persistent=False)
            self.register_buffer("rope_sin", sin, persistent=False)

        self.apply(self._init_weights)
        if config.depth_scaled_init:
            for name, param in self.named_parameters():
                if name.endswith("c_proj.weight"):
                    nn.init.normal_(param, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None, z_loss_weight=0.0):
        seq = idx.size(1)
        if seq > self.config.sequence_length:
            raise ValueError(f"sequence {seq} exceeds context {self.config.sequence_length}")
        x = self.wte(idx)
        if self.wpe is not None:
            x = x + self.wpe(torch.arange(seq, device=idx.device))
        rope = (self.rope_cos[:, :, :seq], self.rope_sin[:, :, :seq]) if self.config.pos_emb == "rope" else None
        x = self.drop(x)
        for block in self.blocks:
            x = block(x, rope)
        logits = self.lm_head(self.ln_f(x)).float()
        if self.config.logit_softcap > 0:
            cap = self.config.logit_softcap
            logits = cap * torch.tanh(logits / cap)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            if z_loss_weight > 0:
                loss = loss + z_loss_weight * torch.logsumexp(logits, dim=-1).square().mean()
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, valid_vocab_size=None):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.sequence_length :]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-6)
            if valid_vocab_size is not None:
                logits[:, valid_vocab_size:] = float("-inf")  # never sample padding ids
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)
        return idx

    def num_params(self, non_embedding=False):
        total = sum(param.numel() for param in self.parameters())
        if non_embedding:
            total -= self.wte.weight.numel()
            if self.wpe is not None:
                total -= self.wpe.weight.numel()
            if not self.config.tie_embeddings:
                total -= self.lm_head.weight.numel()
        return total