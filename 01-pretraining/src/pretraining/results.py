import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # 01-pretraining
SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
MARKERS = ("<!-- results:start -->", "<!-- results:end -->")
ABLATION_MARKERS = ("<!-- ablations:start -->", "<!-- ablations:end -->")
RUN_PATTERN = re.compile(r"^(?P<variant>.+)_lr(?P<lr>[^_]+)(?:_seed(?P<seed>\d+))?$")


def load_run(log_path):
    run = {"name": log_path.parent.name, "args": {}, "params": None, "evals": [], "hellaswag": {}}
    with log_path.open() as file:
        for line in file:
            entry = json.loads(line)
            if "args" in entry:
                run["args"], run["params"] = entry["args"], entry["non_embedding_params"]
            elif "eval" in entry and "val_loss" in entry["eval"]:
                run["evals"].append(entry)
            elif "eval" in entry:
                run["hellaswag"] = entry["eval"]
    return run


def results_table(runs):
    lines = [
        "| Run | Non-emb params | Tokens | Val loss | Val bpb | HellaSwag acc_norm | Hours |",
        "|---|---|---|---|---|---|---|",
    ]
    for run in runs:
        last = run["evals"][-1]
        hellaswag = run["hellaswag"].get("hellaswag_acc_norm")
        lines.append(
            f"| {run['name']} | {run['params'] / 1e6:.1f}M | {last.get('tokens', 0) / 1e9:.2f}B | "
            f"{last['eval']['val_loss']:.4f} | {last['eval']['val_bpb']:.4f} | "
            f"{f'{hellaswag:.2%}' if hellaswag is not None else '-'} | {last.get('elapsed', 0) / 3600:.2f} |"
        )
    return "\n".join(lines)


def ablation_table(runs, baseline):
    groups = {}
    for run in runs:
        match = RUN_PATTERN.match(run["name"])
        if match:
            groups.setdefault(match["variant"], []).append((float(match["lr"]), match["seed"], run))
    if baseline not in groups:
        raise SystemExit(f"baseline {baseline!r} not found, have: {sorted(groups)}")

    def best(entries):
        unseeded = [entry for entry in entries if entry[1] is None]
        return min(unseeded, key=lambda entry: entry[2]["evals"][-1]["eval"]["val_loss"])

    base_lr, _, base_run = best(groups[baseline])
    base_loss = base_run["evals"][-1]["eval"]["val_loss"]
    seeded_lrs = {entry[0] for entry in groups[baseline] if entry[1] is not None}
    seed_lr = seeded_lrs.pop() if seeded_lrs else base_lr
    seed_losses = [entry[2]["evals"][-1]["eval"]["val_loss"] for entry in groups[baseline] if entry[0] == seed_lr]
    noise = max(seed_losses) - min(seed_losses) if len(seed_losses) > 1 else 0.0

    lines = [
        f"Baseline `{baseline}` at lr {base_lr:g}: val loss {base_loss:.4f}. "
        f"Noise floor (seed range at lr {seed_lr:g}, {len(seed_losses)} seeds): {noise:.4f}.",
        "",
        "| Variant | Best LR | LRs tried | Non-emb params | Val loss | Δ vs baseline | HellaSwag acc_norm | Verdict |",
        "|---|---|---|---|---|---|---|---|",
    ]
    rows = []
    for variant, entries in groups.items():
        if not any(entry[1] is None for entry in entries):
            continue
        lr, _, run = best(entries)
        tried = sorted({entry[0] for entry in entries if entry[1] is None})
        loss = run["evals"][-1]["eval"]["val_loss"]
        delta = loss - base_loss
        verdict = "baseline" if variant == baseline else "better" if delta < -noise else "worse" if delta > noise else "tie"
        if len(tried) > 1 and lr in (tried[0], tried[-1]):
            verdict += ", LR at sweep edge"
        hellaswag = run["hellaswag"].get("hellaswag_acc_norm")
        rows.append((loss, f"| {variant} | {lr:g} | {', '.join(f'{value:g}' for value in tried)} | {run['params'] / 1e6:.1f}M | "
                           f"{loss:.4f} | {delta:+.4f} | {f'{hellaswag:.2%}' if hellaswag is not None else '-'} | {verdict} |"))
    lines += [row for _, row in sorted(rows)]
    return "\n".join(lines)


def update_results_md(path, table, markers=MARKERS):
    text = path.read_text()
    block = f"{markers[0]}\n{table}\n{markers[1]}"
    pattern = re.compile(re.escape(markers[0]) + ".*?" + re.escape(markers[1]), re.S)
    text = pattern.sub(lambda match: block, text) if pattern.search(text) else text.rstrip() + "\n\n" + block + "\n"
    path.write_text(text)


def plot_curves(runs, path, skip_fraction):
    import matplotlib.pyplot as plt

    if len(runs) > len(SERIES_COLORS):
        print(f"plotting the first {len(SERIES_COLORS)} of {len(runs)} runs, narrow --runs to choose")
        runs = runs[: len(SERIES_COLORS)]
    fig, axis = plt.subplots(figsize=(8, 4.5), dpi=150, facecolor="#fcfcfb")
    axis.set_facecolor("#fcfcfb")
    for run, color in zip(runs, SERIES_COLORS):
        total = run["evals"][-1].get("tokens", 0)
        points = [entry for entry in run["evals"] if entry.get("tokens", 0) >= skip_fraction * total]
        axis.plot([entry["tokens"] / 1e9 for entry in points], [entry["eval"]["val_loss"] for entry in points],
                  color=color, linewidth=2, label=run["name"])
    axis.set_xlabel("Tokens (B)", color="#52514e")
    axis.set_ylabel("Val loss", color="#52514e")
    axis.set_title("Validation loss", loc="left", color="#0b0b0b")
    axis.grid(color="#e1e0d9", linewidth=0.8)
    axis.tick_params(colors="#52514e")
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    axis.legend(frameon=False, labelcolor="#0b0b0b")
    fig.tight_layout()
    fig.savefig(path)
    print(f"saved {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=str, default="*")  # glob over run names, e.g. "s_*"
    parser.add_argument("--out_dir", type=str, default=str(ROOT / "output"))
    parser.add_argument("--update_results", action="store_true")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--ablations", type=str, default="")  # baseline variant, e.g. "s_adamw"
    parser.add_argument("--skip_fraction", type=float, default=0.1)  # hide the early high-loss part
    args = parser.parse_args()

    runs = [load_run(path) for path in sorted(Path(args.out_dir).glob(f"{args.runs}/log.jsonl"))]
    runs = [run for run in runs if run["evals"]]
    if not runs:
        raise SystemExit(f"no runs matching {args.runs!r} in {args.out_dir}")
    table = results_table(runs)
    print(table)
    if args.update_results:
        update_results_md(ROOT / "RESULTS.md", table)
    if args.ablations:
        prefix = args.ablations.split("_")[0] + "_"
        ablations = ablation_table([run for run in runs if run["name"].startswith(prefix)], args.ablations)
        print("\n" + ablations)
        if args.update_results:
            update_results_md(ROOT / "studies/architecture/RESULTS.md", ablations, ABLATION_MARKERS)
    if args.plot:
        plot_curves(runs, Path(args.out_dir) / f"val_loss_{args.runs.replace('*', 'all')}.png", args.skip_fraction)


if __name__ == "__main__":
    main()