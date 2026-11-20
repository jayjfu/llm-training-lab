import argparse
import json
import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
LEADERBOARD_COLUMNS = [
    ("leaderboard_ifeval", "IFEval", ["prompt_level_strict_acc,none", "inst_level_strict_acc,none"]),
    ("leaderboard_musr", "MuSR", ["acc_norm,none"]),
    ("leaderboard_math_hard", "MATH Lvl-5", ["exact_match,none"]),
    ("leaderboard_bbh", "BBH", ["acc_norm,none"]),
    ("leaderboard_mmlu_pro", "MMLU-Pro", ["acc,none"]),
    ("leaderboard_gpqa", "GPQA", ["acc_norm,none"]),
]
CODING_COLUMNS = [
    ("humaneval_plus", "HumanEval", ["pass@1_base,none"]),
    ("humaneval_plus", "HumanEval+", ["pass@1,none"]),
    ("mbpp_plus", "MBPP", ["pass@1_base,none"]),
    ("mbpp_plus", "MBPP+", ["pass@1,none"]),
]
TABLES = {"results": LEADERBOARD_COLUMNS, "coding": CODING_COLUMNS}
STAGES = ["base", "sft", "preference", "rl", "distillation", "instruct"]
STAGE_LABELS = {"sft": "SFT", "rl": "RL"}
MODEL_NAMES = {"base": "Qwen/Qwen3.5-0.8B-Base", "instruct": "Qwen/Qwen3.5-0.8B"}


def metric_value(results, task, metric):
    if metric in results.get(task, {}):
        return results[task][metric]
    subtask_values = [scores[metric] for name, scores in results.items() if name.startswith(task + "_") and metric in scores]
    return sum(subtask_values) / len(subtask_values) if subtask_values else None


def task_score(run_dir, task, metrics):
    result_files = sorted((run_dir / task).rglob("results_*.json"))
    if not result_files:
        return None
    results = json.loads(result_files[-1].read_text())["results"]
    values = [metric_value(results, task, metric) for metric in metrics]
    return None if None in values else 100 * sum(values) / len(values)


def run_stage(run, ckpts_dir):
    if run in MODEL_NAMES:
        return run
    run_config = ckpts_dir / run / "run.yaml"
    return yaml.safe_load(run_config.read_text()).get("stage", "other") if run_config.exists() else "other"


def collect(results_dir, ckpts_dir, columns):
    rows = []
    for run_dir in sorted(path for path in results_dir.iterdir() if path.is_dir()):
        scores = [task_score(run_dir, task, metrics) for task, _, metrics in columns]
        if all(score is None for score in scores):
            continue
        average = None if None in scores else sum(scores) / len(scores)
        rows.append({"run": run_dir.name, "stage": run_stage(run_dir.name, ckpts_dir), "scores": scores + [average]})
    return sorted(rows, key=lambda row: (STAGES.index(row["stage"]) if row["stage"] in STAGES else len(STAGES), row["run"]))


def render(rows, columns):
    base = next((row["scores"] for row in rows if row["run"] == "base"), None)
    header = ["Stage", "Model"] + [column for _, column, _ in columns] + ["Avg"]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for row in rows:
        cells = []
        for index, score in enumerate(row["scores"]):
            if score is None:
                cells.append("-")
            elif base is None or row["run"] == "base" or base[index] is None:
                cells.append(f"{score:.2f}")
            else:
                cells.append(f"{score:.2f} ({score - base[index]:+.2f})")
        model = MODEL_NAMES.get(row["run"], row["run"])
        lines.append(f"| **{STAGE_LABELS.get(row['stage'], row['stage'].capitalize())}** | {model} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def best_per_stage(rows):
    best = {}
    for row in rows:
        average = row["scores"][-1]
        if average is not None and (row["stage"] not in best or average > best[row["stage"]]["scores"][-1]):
            best[row["stage"]] = row
    return [row for row in rows if best.get(row["stage"]) is row]


def replace_block(path, table, marker):
    text = path.read_text()
    pattern = rf"(<!-- {marker}:start -->\n).*?(<!-- {marker}:end -->)"
    path.write_text(re.sub(pattern, lambda match: match.group(1) + table + "\n" + match.group(2), text, flags=re.S))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, default=str(ROOT / "03-post-training/results"))
    parser.add_argument("--ckpts_dir", type=str, default=str(ROOT / "03-post-training/ckpts"))
    args = parser.parse_args()

    for marker, columns in TABLES.items():
        rows = collect(Path(args.results_dir), Path(args.ckpts_dir), columns)
        replace_block(ROOT / "03-post-training/RESULTS.md", render(rows, columns), marker)
        replace_block(ROOT / "03-post-training/README.md", render(best_per_stage(rows), columns), marker)
        replace_block(ROOT / "README.md", render(best_per_stage(rows), columns), marker)
        print(render(rows, columns))


if __name__ == "__main__":
    main()