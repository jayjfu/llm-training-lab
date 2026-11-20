import argparse
import re
import statistics
from pathlib import Path

import tiktoken
import torch

from fast_inference.generate import generate_cached, generate_cuda_graph, generate_naive, make_decode_step
from fast_inference.model import load_model

ROOT = Path(__file__).resolve().parents[2]  # 05-optimization/inference
REPO = Path(__file__).resolve().parents[4]
PROMPT_TEXT = "The history of science is the study of the development of science and scientific knowledge. "


def strategies(model, valid_vocab_size):
    compiled_step = torch.compile(make_decode_step(model, valid_vocab_size), fullgraph=True)
    graph_step = torch.compile(make_decode_step(model, valid_vocab_size), fullgraph=True, mode="max-autotune-no-cudagraphs")
    return {
        "naive": lambda prompt, new: generate_naive(model, prompt, new, valid_vocab_size),
        "kv_cache": lambda prompt, new: generate_cached(model, prompt, new, valid_vocab_size, "dynamic"),
        "static_cache": lambda prompt, new: generate_cached(model, prompt, new, valid_vocab_size, "static"),
        "compile": lambda prompt, new: generate_cached(model, prompt, new, valid_vocab_size, "static", compiled_step),
        "cuda_graph": lambda prompt, new: generate_cuda_graph(model, prompt, new, valid_vocab_size),
        "compile_graph": lambda prompt, new: generate_cuda_graph(model, prompt, new, valid_vocab_size, graph_step),
    }


def update_results_md(path, block, tag):
    """One marked table per tag (e.g. batch1, batch64), so runs with different settings don't overwrite each other."""
    markers = (f"<!-- {tag}:start -->", f"<!-- {tag}:end -->")
    text = path.read_text() if path.exists() else "# Inference Results\n"
    block = f"{markers[0]}\n{block}\n{markers[1]}"
    pattern = re.compile(re.escape(markers[0]) + ".*?" + re.escape(markers[1]), re.S)
    text = pattern.sub(lambda match: block, text) if pattern.search(text) else text.rstrip() + "\n\n" + block + "\n"
    path.write_text(text)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default=str(REPO / "01-pretraining/ckpts/124m_baseline/latest.pt"))
    parser.add_argument("--modes", nargs="+", default=["naive", "kv_cache", "static_cache", "compile", "cuda_graph", "compile_graph"])
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--prompt_len", type=int, default=128)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--update_results", action="store_true")
    parser.add_argument("--tag", type=str, default="")  # RESULTS.md table name, default batch<N>
    args = parser.parse_args()

    device = torch.device(args.device)
    model, _ = load_model(args.ckpt, device, torch.bfloat16 if args.dtype == "bf16" else torch.float32)
    enc = tiktoken.get_encoding("gpt2")
    assert args.prompt_len + args.max_new_tokens <= model.config.sequence_length, "prompt + new tokens exceed context"
    prompt_ids = (enc.encode_ordinary(PROMPT_TEXT) * args.prompt_len)[: args.prompt_len]
    prompt = torch.tensor([prompt_ids] * args.batch_size, device=device)

    runners = strategies(model, enc.n_vocab)
    rows, reference = [], None
    for mode in args.modes:
        if mode.endswith("graph") and device.type != "cuda":
            print(f"skip {mode} (needs CUDA)")
            continue
        if device.type == "cuda":
            torch.cuda.empty_cache()  # previous mode's cached blocks must not count here
        runners[mode](prompt, args.max_new_tokens)  # warmup: compile, capture, allocator
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        timings = [runners[mode](prompt, args.max_new_tokens) for _ in range(args.repeats)]
        tokens = timings[-1][0]
        ttft = statistics.median(timing[1] for timing in timings)
        decode = statistics.median(timing[2] for timing in timings)
        reference = tokens if reference is None else reference
        match = (tokens == reference).float().mean().item()
        decode_steps = args.max_new_tokens - 1
        peak_mb = f"{torch.cuda.max_memory_allocated() / 2**20:,.0f}" if device.type == "cuda" else "-"
        rows.append(f"| {mode} | {ttft * 1e3:.1f} | {decode / decode_steps * 1e3:.2f} | "
                    f"{args.batch_size * decode_steps / decode:,.0f} | {peak_mb} | {match:.1%} |")
        print(rows[-1])
    print("sample:", repr(enc.decode(tokens[0, :48].tolist())))

    device_name = torch.cuda.get_device_name() if device.type == "cuda" else "CPU"
    table = "\n".join([
        f"{Path(args.ckpt).parent.name}, {args.dtype}, batch {args.batch_size}, prompt {args.prompt_len}, "
        f"{args.max_new_tokens} new tokens, {device_name}. Median of {args.repeats} runs after warmup.",
        "",
        "| Mode | TTFT (ms) | Decode (ms/token) | Decode (tokens/s) | Peak memory (MB) | Match vs first mode |",
        "|---|---|---|---|---|---|",
        *rows,
    ])
    print("\n" + table)
    if args.update_results:
        update_results_md(ROOT / "RESULTS.md", table, args.tag or f"batch{args.batch_size}")


if __name__ == "__main__":
    main()