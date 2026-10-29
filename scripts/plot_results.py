#!/usr/bin/env python
"""Figures for docs/report: per-task scores, the few-shot budget sweep and the routing weights."""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wemoe_nlp.reporting import majority_validation  # noqa: E402

TASKS = ["mrpc", "rte", "cola", "sst2", "qnli"]
TASK_NAMES = {"mrpc": "MRPC", "rte": "RTE", "cola": "CoLA", "sst2": "SST-2", "qnli": "QNLI"}
TASK_LABELS = {"mrpc": "MRPC\n(acc+F1)/2", "rte": "RTE\nacc", "cola": "CoLA\nMCC", "sst2": "SST-2\nacc",
               "qnli": "QNLI\nacc"}

# Reference palette (dataviz skill): categorical slots in fixed order, chart ink, diverging blue <-> red.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK, INK_2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
DIVERGING = LinearSegmentedColormap.from_list(
    "blue_gray_red", ["#a32a2a", "#e34948", "#f0efec", "#3987e5", "#184f95"])

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 9, "axes.edgecolor": AXIS, "axes.labelcolor": INK_2,
    "axes.titlesize": 10, "axes.titlecolor": INK, "xtick.color": INK_2, "ytick.color": INK_2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "axes.grid.axis": "y",
    "grid.color": GRID, "grid.linewidth": 0.6, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "legend.frameon": False, "legend.fontsize": 8,
})


def load(path: Path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def score_matrix(per_seed: list) -> np.ndarray:
    """(seeds, tasks) task scores ×100."""
    return 100 * np.array([[r[t]["score"] for t in TASKS] for r in per_seed])


def wemoe_groups(results_dir: Path) -> dict:
    groups = defaultdict(list)
    for path in sorted((results_dir / "wemoe").glob("*/results.json")):
        run = load(path)
        cfg = run["fsft_config"]
        groups[(cfg["moe_scope"], cfg["router_hidden_layers"], cfg["k"])].append(run)
    return groups


def plot_per_task(baselines: dict, majority, groups: dict, out: Path) -> None:
    series = [("Task Arithmetic", score_matrix([baselines["task_arithmetic"]["validation"]]))]
    for key, label in (((("full", 2, 256)), "WEMoE + FSFT, full MLP"),
                       (("intermediate", 2, 256), "WEMoE + FSFT, intermediate"),
                       (("full", 0, 256), "Layer-wise coefficients (h=0)")):
        if key in groups:
            series.append((label, score_matrix([r["validation"] for r in groups[key]])))
    expert = score_matrix([baselines["single_task"]["validation"]])[0]

    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    x = np.arange(len(TASKS))
    width = 0.8 / len(series)
    for i, (label, scores) in enumerate(series):
        pos = x - 0.4 + width * (i + 0.5)
        yerr = scores.std(axis=0, ddof=1) if len(scores) > 1 else None
        ax.bar(pos, scores.mean(axis=0), width - 0.03, color=SERIES[i], label=label, zorder=2)
        if yerr is not None:
            ax.errorbar(pos, scores.mean(axis=0), yerr=yerr, fmt="none", ecolor=INK_2, elinewidth=1, capsize=2,
                        zorder=3)
    ax.hlines(expert, x - 0.45, x + 0.45, colors=INK, linewidth=2, label="Single-task expert", zorder=4)
    if majority:
        ax.hlines(score_matrix([majority])[0], x - 0.45, x + 0.45, colors=MUTED, linewidth=1.5,
                  linestyles=(0, (3, 2)), label="Majority class", zorder=4)
    ax.set_xticks(x, [TASK_LABELS[t] for t in TASKS])
    ax.set_ylabel("Validation score (×100)")
    ax.set_ylim(0, 100)
    ax.tick_params(axis="x", length=0)
    ax.set_title("Per-task validation scores (k = 256, mean ± std over seeds)", loc="left")
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.2))
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def plot_k_sweep(baselines: dict, majority, groups: dict, out: Path) -> None:
    ks = sorted(k for scope, layers, k in groups if scope == "full" and layers == 2)
    if not ks:
        return
    per_seed = {k: [np.mean(score_matrix([r["validation"]])) for r in groups[("full", 2, k)]] for k in ks}

    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    pos = np.arange(len(ks))
    for i, k in enumerate(ks):
        ax.scatter(np.full(len(per_seed[k]), pos[i]), per_seed[k], s=22, color=SERIES[0], alpha=0.35,
                   linewidths=0, zorder=3)
    means = [np.mean(per_seed[k]) for k in ks]
    ax.plot(pos, means, color=SERIES[0], linewidth=2, marker="o", markersize=7, zorder=4,
            markeredgecolor=SURFACE, markeredgewidth=1.5)
    for i, m in enumerate(means):
        ax.annotate(f"{m:.1f}", (pos[i], m), textcoords="offset points", xytext=(10, -12), color=INK, fontsize=8)
    references = [("Single-task experts", baselines["single_task"]["validation"], INK, "-"),
                  ("Task Arithmetic", baselines["task_arithmetic"]["validation"], INK_2, (0, (5, 2)))]
    if majority:
        references.append(("Majority class", majority, MUTED, (0, (2, 2))))
    for label, metrics, color, style in references:
        value = np.mean(score_matrix([metrics]))
        ax.axhline(value, color=color, linewidth=1.2, linestyle=style, zorder=2)
        ax.annotate(f"{label} {value:.1f}", (pos[-1] + 0.15, value), textcoords="offset points", xytext=(0, 3),
                    color=color, fontsize=8, ha="right")
    ax.set_xticks(pos, [f"k = {k}" for k in ks])
    ax.set_xlim(-0.3, len(ks) - 0.6)
    ax.set_ylim(40, 82)
    ax.set_ylabel("Average validation score (×100)")
    ax.set_title("WEMoE + FSFT (full MLP): examples per task vs. score\nline = mean, dots = seeds",
                 loc="left")
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def plot_routing(predictions: Path, out: Path, title: str) -> None:
    routing = defaultdict(list)
    with open(predictions, encoding="utf-8") as fh:
        for line in fh:
            record = json.loads(line)
            routing[record["task"]].append(record["routing"])
    means = {t: np.mean(np.array(routing[t]), axis=0) for t in TASKS}  # (layers, experts)
    limit = max(np.abs(m).max() for m in means.values())
    norm = TwoSlopeNorm(vcenter=0.0, vmin=-limit, vmax=limit)

    fig, axes = plt.subplots(1, len(TASKS), figsize=(9.6, 3.6), sharey=True)
    for ax, task in zip(axes, TASKS):
        image = ax.imshow(means[task], cmap=DIVERGING, norm=norm, aspect="auto")
        ax.set_title(f"{TASK_NAMES[task]} inputs", fontsize=9)
        ax.set_xticks(range(len(TASKS)), [TASK_NAMES[t] for t in TASKS], rotation=90, fontsize=7)
        ax.set_yticks(range(means[task].shape[0]), [str(i + 1) for i in range(means[task].shape[0])], fontsize=7)
        ax.grid(False)
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
    axes[0].set_ylabel("Layer")
    fig.supxlabel("Expert task vector", fontsize=9, color=INK_2, y=-0.08)
    fig.suptitle(title, x=0.01, ha="left", fontsize=10, color=INK)
    bar = fig.colorbar(image, ax=axes, fraction=0.025, pad=0.02)
    bar.set_label("Mean routing weight (unnormalised; ≈ λ at init)", color=INK_2)
    bar.outline.set_visible(False)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--out-dir", default="docs/report/figures")
    parser.add_argument("--routing-run", default="full-mean-k256-s0")
    args = parser.parse_args()
    results_dir, out_dir = Path(args.results_dir), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    baselines = load(results_dir / "baselines.json")
    groups = wemoe_groups(results_dir)
    majority = majority_validation(results_dir, TASKS, baselines)
    plot_per_task(baselines, majority, groups, out_dir / "per_task.png")
    plot_k_sweep(baselines, majority, groups, out_dir / "k_sweep.png")
    predictions = results_dir / "wemoe" / args.routing_run / "predictions_validation.jsonl"
    if predictions.exists():
        plot_routing(predictions, out_dir / "routing.png",
                     f"Mean routing weights per layer, by input task ({args.routing_run})")
    print(f"Figures in {out_dir}")


if __name__ == "__main__":
    main()
