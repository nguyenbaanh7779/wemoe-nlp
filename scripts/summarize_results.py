#!/usr/bin/env python
"""Collect results/*.json into Markdown tables (mean ± std over seeds) in results/summary.md."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

TASK_ORDER = ["mrpc", "rte", "cola", "sst2", "qnli"]
TASK_HEADERS = {"mrpc": "MRPC (acc/F1)", "rte": "RTE (acc)", "cola": "CoLA (MCC)", "sst2": "SST-2 (acc)",
                "qnli": "QNLI (acc)"}
TASK_METRICS = {"mrpc": ["accuracy", "f1"], "cola": ["mcc"]}


def load(path: Path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def fmt(values) -> str:
    values = 100 * np.asarray(values, dtype=float)
    if len(values) == 1:
        return f"{values[0]:.1f}"
    return f"{values.mean():.1f}±{values.std(ddof=1):.1f}"


def normalized(run: dict, tasks: list, single: dict):
    """Mean of task score / single-task score, skipping tasks whose single-task score is 0."""
    ratios = [run[t]["score"] / single[t]["score"] for t in tasks if single[t]["score"]]
    return float(np.mean(ratios)) if ratios else None


def row(name: str, runs: list, tasks: list, single: dict) -> str:
    """``runs``: one or more per-task metric dicts (one per seed)."""
    cells = [name]
    for task in tasks:
        metrics = TASK_METRICS.get(task, ["accuracy"])
        cells.append("/".join(fmt([r[task][m] for r in runs]) for m in metrics))
    cells.append(fmt([np.mean([r[t]["score"] for t in tasks]) for r in runs]))
    ratios = [normalized(r, tasks, single) for r in runs] if single else [None]
    cells.append("-" if None in ratios else fmt(ratios))
    return "| " + " | ".join(cells) + " |"


def table(rows: list, tasks: list) -> list:
    header = "| Model | " + " | ".join(TASK_HEADERS.get(t, t) for t in tasks) + " | Avg score | Normalized |"
    return [header, "|---|" + "---:|" * (len(tasks) + 2)] + rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args()
    results_dir = Path(args.results_dir)

    baselines = load(results_dir / "baselines.json") if (results_dir / "baselines.json").exists() else None
    runs = [load(p) for p in sorted((results_dir / "wemoe").glob("*/results.json"))]
    tasks = baselines["tasks"] if baselines else (runs[0]["tasks"] if runs else TASK_ORDER)
    single = baselines["single_task"]["validation"] if baselines else None

    groups = defaultdict(list)
    for run in runs:
        cfg = run["fsft_config"]
        key = (cfg["moe_scope"], cfg["routing"], cfg["k"], cfg["router_hidden_layers"], cfg["lr"], cfg["max_steps"])
        groups[key].append(run)

    lines = ["# Results (official GLUE validation splits)", ""]
    main_rows = []
    if baselines:
        main_rows.append(row("Single-task experts", [single], tasks, single))
        main_rows.append(row(f"Simple Averaging (λ=1/{len(tasks)})", [baselines["simple_average"]["validation"]],
                             tasks, single))
        lam = baselines["task_arithmetic"]["best_lambda"]
        main_rows.append(row(f"Task Arithmetic (λ={lam})", [baselines["task_arithmetic"]["validation"]], tasks, single))
    for (scope, routing, k, layers, lr, steps), members in sorted(groups.items()):
        seeds = sorted(m["fsft_config"]["seed"] for m in members)
        main_rows.append(row(f"WEMoE init ({scope}, λ={members[0]['init_lambda']})",
                             [m["init_validation"] for m in members], tasks, single))
        main_rows.append(row(f"WEMoE + FSFT ({scope}, {routing}, k={k}, seeds={seeds})",
                             [m["validation"] for m in members], tasks, single))
    lines += table(main_rows, tasks)
    lines += ["", "Scores ×100; mean±std over seeds. Avg score: mean of task scores (MRPC: mean of acc and F1, "
              "CoLA: MCC). Normalized: task score / single-task score, averaged over tasks.", ""]

    efficiency_path = results_dir / "efficiency.json"
    if efficiency_path.exists():
        eff = load(efficiency_path)
        lines += ["## Parameters and storage", "", "| Deployment | # checkpoints | Stored params | vs 5 models |",
                  "|---|---:|---:|---:|"]
        for key in ("single_task_models", "task_arithmetic", "wemoe_full", "wemoe_intermediate"):
            r = eff[key]
            lines.append(f"| {key} | {r['num_checkpoints']} | {r['stored_params'] / 1e6:.1f}M | "
                         f"{100 * r['fraction_of_single_task_models']:.1f}% |")
        lines.append("")

    text = "\n".join(lines)
    (results_dir / "summary.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
