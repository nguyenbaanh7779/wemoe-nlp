#!/usr/bin/env python
"""Collect results/*.json into Markdown tables (mean ± std over seeds) in results/summary.md."""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wemoe_nlp.efficiency import efficiency_markdown  # noqa: E402
from wemoe_nlp.metrics import gain_over_majority, mean_score, normalized_scores  # noqa: E402
from wemoe_nlp.reporting import majority_validation  # noqa: E402

TASK_ORDER = ["mrpc", "rte", "cola", "sst2", "qnli"]
TASK_HEADERS = {"mrpc": "MRPC (acc/F1)", "rte": "RTE (acc)", "cola": "CoLA (MCC)", "sst2": "SST-2 (acc)",
                "qnli": "QNLI (acc)"}
TASK_METRICS = {"mrpc": ["accuracy", "f1"], "cola": ["mcc"]}
DEFAULT_ROUTER_LAYERS = 2


def load(path: Path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def fmt(values) -> str:
    values = 100 * np.asarray(values, dtype=float)
    if len(values) == 1:
        return f"{values[0]:.1f}"
    return f"{values.mean():.1f}±{values.std(ddof=1):.1f}"


def row(name: str, runs: list, tasks: list, single, majority) -> str:
    """``runs``: one or more per-task metric dicts (one per seed)."""
    cells = [name]
    for task in tasks:
        metrics = TASK_METRICS.get(task, ["accuracy"])
        cells.append("/".join(fmt([r[task][m] for r in runs]) for m in metrics))
    runs = [{t: r[t] for t in tasks} for r in runs]
    cells.append(fmt([mean_score(r) for r in runs]))
    cells.append(fmt_means([normalized_scores(r, single) for r in runs]) if single else "-")
    cells.append(fmt_means([gain_over_majority(r, single, majority) for r in runs]) if majority else "-")
    return "| " + " | ".join(cells) + " |"


def fmt_means(per_run: list) -> str:
    """Format the "mean" entries; "-" when a run has none (every task skipped, e.g. in smoke runs)."""
    means = [r.get("mean") for r in per_run]
    return "-" if None in means else fmt(means)


def table(rows: list, tasks: list) -> list:
    header = ("| Model | " + " | ".join(TASK_HEADERS.get(t, t) for t in tasks)
              + " | Avg score | Normalized | Gain over majority |")
    return [header, "|---|" + "---:|" * (len(tasks) + 3)] + rows


def router_label(scope: str, routing: str, layers: int) -> str:
    parts = [scope, routing]
    if layers != DEFAULT_ROUTER_LAYERS:
        parts.append(f"h={layers}")
    return ", ".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args()
    results_dir = Path(args.results_dir)

    baselines = load(results_dir / "baselines.json") if (results_dir / "baselines.json").exists() else None
    runs = [load(p) for p in sorted((results_dir / "wemoe").glob("*/results.json"))]
    tasks = baselines["tasks"] if baselines else (runs[0]["tasks"] if runs else TASK_ORDER)
    single = baselines["single_task"]["validation"] if baselines else None
    majority = majority_validation(results_dir, tasks, baselines) if single else None

    # WEMoE at init only depends on the router architecture and the seed, not on k: one row per architecture
    inits = defaultdict(dict)
    groups = defaultdict(list)
    for run in runs:
        cfg = run["fsft_config"]
        arch = (cfg["moe_scope"], cfg["routing"], cfg["router_hidden_layers"])
        inits[arch][cfg["seed"]] = run
        groups[arch + (cfg["k"], cfg["lr"], cfg["max_steps"])].append(run)
    varying = {name: len({run["fsft_config"][name] for run in runs}) > 1 for name in ("lr", "max_steps")}

    lines = ["# Results (official GLUE validation splits)", ""]
    main_rows = []
    if baselines:
        main_rows.append(row("Single-task experts", [single], tasks, single, majority))
        if majority:
            main_rows.append(row("Majority class", [majority], tasks, single, majority))
        main_rows.append(row(f"Simple Averaging (λ=1/{len(tasks)})", [baselines["simple_average"]["validation"]],
                             tasks, single, majority))
        lam = baselines["task_arithmetic"]["best_lambda"]
        main_rows.append(row(f"Task Arithmetic (λ={lam})", [baselines["task_arithmetic"]["validation"]], tasks,
                             single, majority))
    shown_inits = set()
    for key, members in sorted(groups.items()):
        scope, routing, layers, k, lr, steps = key
        arch = (scope, routing, layers)
        label = router_label(scope, routing, layers)
        if arch not in shown_inits:
            shown_inits.add(arch)
            by_seed = inits[arch]
            main_rows.append(row(f"WEMoE init ({label}, λ={members[0]['init_lambda']}, seeds={sorted(by_seed)})",
                                 [by_seed[s]["init_validation"] for s in sorted(by_seed)], tasks, single, majority))
        extra = (f", lr={lr}" if varying["lr"] else "") + (f", steps={steps}" if varying["max_steps"] else "")
        seeds = sorted(m["fsft_config"]["seed"] for m in members)
        main_rows.append(row(f"WEMoE + FSFT ({label}, k={k}{extra}, seeds={seeds})",
                             [m["validation"] for m in members], tasks, single, majority))
    lines += table(main_rows, tasks)
    lines += ["", "Scores ×100; mean±std over seeds. Avg score: mean of task scores (MRPC: mean of acc and F1, "
              "CoLA: MCC). Normalized: task score / single-task score, averaged over tasks. Gain over majority: "
              "(score − majority) / (single-task − majority), averaged over tasks; 0 is no better than always "
              "predicting the most frequent label.", ""]

    expert_runs = {t: load(results_dir / "experts" / f"{t}.json") for t in tasks
                   if (results_dir / "experts" / f"{t}.json").exists()}
    if expert_runs or groups:
        lines += ["## Training cost", "", "| Run | Best step / epoch | Train time (min) |", "|---|---:|---:|"]
        for task, result in expert_runs.items():
            lines.append(f"| Expert {task} | epoch {result['best_epoch']} | {result['train_seconds'] / 60:.1f} |")
        for (scope, routing, layers, k, lr, steps), members in sorted(groups.items()):
            best = [m["best_step"] for m in members]
            minutes = np.mean([m["train_seconds"] for m in members]) / 60
            lines.append(f"| WEMoE ({router_label(scope, routing, layers)}, k={k}) | "
                         f"{', '.join(map(str, best))} of {steps} | {minutes:.1f} |")
        lines += ["", "Train time includes the dev evaluations; WEMoE times are means over seeds.", ""]

    efficiency_path = results_dir / "efficiency.json"
    if efficiency_path.exists():
        lines += ["## Parameters, storage and latency", ""] + efficiency_markdown(load(efficiency_path)) + [""]

    text = "\n".join(lines)
    (results_dir / "summary.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
