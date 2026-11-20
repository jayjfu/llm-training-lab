"""Greedy decoding, from naive recompute to CUDA graphs. Each strategy returns (tokens, ttft_seconds, decode_seconds)."""
import time

import torch


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def greedy(logits, valid_vocab_size):
    logits = logits[:, -1, :valid_vocab_size]  # never pick padding ids
    return logits.argmax(dim=-1, keepdim=True)


@torch.no_grad()
def generate_naive(model, prompt, max_new_tokens, valid_vocab_size):
    """Step 0: recompute the whole sequence for every new token."""
    model.setup_cache("none")
    tokens = prompt
    start = time.perf_counter()
    for step in range(max_new_tokens):
        positions = torch.arange(tokens.size(1), device=tokens.device)
        tokens = torch.cat([tokens, greedy(model(tokens, positions, last_only=True), valid_vocab_size)], dim=1)
        if step == 0:
            sync(tokens.device)
            first = time.perf_counter()
    sync(tokens.device)
    return tokens[:, prompt.size(1):], first - start, time.perf_counter() - first


def make_decode_step(model, valid_vocab_size):
    def decode_step(token, position):
        return greedy(model(token, position), valid_vocab_size)
    return decode_step


@torch.no_grad()
def generate_cached(model, prompt, max_new_tokens, valid_vocab_size, cache_mode, decode_step=None):
    """Steps 1-2: prefill once, then decode one token at a time from the KV cache."""
    batch_size, prompt_len = prompt.shape
    model.setup_cache(cache_mode, batch_size, prompt_len + max_new_tokens)
    decode_step = decode_step or make_decode_step(model, valid_vocab_size)
    start = time.perf_counter()
    token = greedy(model(prompt, torch.arange(prompt_len, device=prompt.device), last_only=True), valid_vocab_size)
    sync(prompt.device)
    first = time.perf_counter()
    generated = [token]
    positions = torch.arange(prompt_len, prompt_len + max_new_tokens, device=prompt.device)  # no host-to-device copy per step
    for step in range(max_new_tokens - 1):
        token = decode_step(token, positions[step : step + 1])
        generated.append(token)
    sync(prompt.device)
    return torch.cat(generated, dim=1), first - start, time.perf_counter() - first


class GraphDecoder:
    """Step 3: capture one decode step as a CUDA graph and replay it, removing per-kernel launch overhead."""

    def __init__(self, model, batch_size, valid_vocab_size, decode_step=None, warmup_steps=3):
        device = model.lm_head.weight.device
        self.token = torch.zeros((batch_size, 1), dtype=torch.long, device=device)
        self.position = torch.zeros(1, dtype=torch.long, device=device)
        decode_step = decode_step or make_decode_step(model, valid_vocab_size)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(warmup_steps):  # writes junk at position 0, overwritten by prefill
                decode_step(self.token, self.position)
        torch.cuda.current_stream().wait_stream(stream)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.next_token = decode_step(self.token, self.position)

    def __call__(self, token, position):
        self.token.copy_(token)
        self.position.copy_(position)
        self.graph.replay()
        return self.next_token.clone()


@torch.no_grad()
def generate_cuda_graph(model, prompt, max_new_tokens, valid_vocab_size, decode_step=None):
    """Step 3: eager decode step in a graph. With a compiled decode_step: fused kernels in a graph (gpt-fast)."""
    batch_size, prompt_len = prompt.shape
    model.setup_cache("static", batch_size, prompt_len + max_new_tokens)
    decoder = GraphDecoder(model, batch_size, valid_vocab_size, decode_step)  # capture before prefill: graphs bind cache memory
    start = time.perf_counter()
    token = greedy(model(prompt, torch.arange(prompt_len, device=prompt.device), last_only=True), valid_vocab_size)
    sync(prompt.device)
    first = time.perf_counter()
    generated = [token]
    positions = torch.arange(prompt_len, prompt_len + max_new_tokens, device=prompt.device)
    for step in range(max_new_tokens - 1):
        token = decoder(token, positions[step : step + 1])
        generated.append(token)
    sync(prompt.device)
    return torch.cat(generated, dim=1), first - start, time.perf_counter() - first