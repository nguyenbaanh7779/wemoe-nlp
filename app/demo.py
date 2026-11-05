#!/usr/bin/env python
"""Interactive WEMoE demo: one input, three models (expert, Task Arithmetic, WEMoE) and the routing weights.

    pip install -r requirements-demo.txt
    python app/demo.py                      # http://127.0.0.1:7860
    python app/demo.py --share              # also a public link (Colab)
"""

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import gradio as gr  # noqa: E402

from examples import EXAMPLES, example_title  # noqa: E402
from wemoe_nlp.data import DEFAULT_TASKS  # noqa: E402
from wemoe_nlp.demo import DemoBackend, default_device, demo_runs  # noqa: E402

TASK_NAMES = {"mrpc": "MRPC", "rte": "RTE", "cola": "CoLA", "sst2": "SST-2", "qnli": "QNLI"}
TASK_QUESTIONS = {
    "mrpc": "Are the two sentences paraphrases of each other?",
    "rte": "Does the premise entail the hypothesis?",
    "cola": "Is the sentence grammatically acceptable English?",
    "sst2": "Is the movie-review sentence positive or negative?",
    "qnli": "Does the sentence contain the answer to the question?",
}
FIELD_LABELS = {
    "mrpc": ("Sentence 1", "Sentence 2"),
    "rte": ("Premise", "Hypothesis"),
    "cola": ("Sentence",),
    "sst2": ("Sentence",),
    "qnli": ("Question", "Sentence"),
}
DEFAULT_TASK = "sst2"

# Same diverging blue <-> red scale as docs/report/figures/routing.png
INK, INK_2, SURFACE, HIGHLIGHT = "#0b0b0b", "#52514e", "#fcfcfb", "#eda100"
DIVERGING = LinearSegmentedColormap.from_list("blue_gray_red", ["#a32a2a", "#e34948", "#f0efec", "#3987e5", "#184f95"])

INTRO = """
# WEMoE: five BERT experts merged into one model

Five BERT-base models were fine-tuned separately on five GLUE tasks and then merged.
For one input, this demo compares three models that serve all five tasks:

- **Single-task expert**: the model fine-tuned on the selected task only (the upper bound).
- **Task Arithmetic**: one static merge, θ₀ + λ·Σ τₜ, the same weights for every input.
- **WEMoE**: in every layer, a router reads the input and mixes the experts' MLP task vectors with its own weights.
  The routers were trained on 256 labelled examples per task (FSFT).

The task only selects the classification head. The routers never see it: they decide from the text alone.
"""


def pretty(label: str) -> str:
    return label.replace("_", " ")


def routing_figure(routing: np.ndarray, experts, task: str, lam: float):
    """Heatmap of the routing weights of one input: layers (rows) x expert task vectors (columns)."""
    layers, num_experts = routing.shape
    limit = max(float(np.abs(routing).max()), 0.5)
    fig, ax = plt.subplots(figsize=(5.4, 5.6))
    fig.patch.set_facecolor(SURFACE)
    image = ax.imshow(routing, cmap=DIVERGING, norm=TwoSlopeNorm(vcenter=0.0, vmin=-limit, vmax=limit), aspect="auto")
    for i in range(layers):
        for j in range(num_experts):
            value = routing[i, j]
            color = "white" if abs(value) > 0.6 * limit else INK
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=7.5, color=color)
    if task in experts:
        column = list(experts).index(task)
        ax.add_patch(Rectangle((column - 0.5, -0.5), 1, layers, fill=False, edgecolor=HIGHLIGHT, linewidth=2.5))
    ax.set_xticks(range(num_experts), [TASK_NAMES.get(e, e) for e in experts], fontsize=8.5, color=INK_2)
    ax.set_yticks(range(layers), [str(i + 1) for i in range(layers)], fontsize=8, color=INK_2)
    ax.set_xlabel("Expert task vector (column of the selected task outlined)", fontsize=8.5, color=INK_2)
    ax.set_ylabel("Layer", fontsize=8.5, color=INK_2)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(f"Routing weights for this input (Task Arithmetic uses {lam:g} everywhere)",
                 fontsize=9, color=INK, loc="left")
    bar = fig.colorbar(image, ax=ax, fraction=0.05, pad=0.03)
    bar.outline.set_visible(False)
    bar.ax.tick_params(labelsize=7, colors=INK_2)
    fig.tight_layout()
    return fig


def routing_summary(routing: np.ndarray, experts, task: str) -> str:
    means = routing.mean(axis=0)
    order = np.argsort(-means)
    ranked = ", ".join(f"{TASK_NAMES.get(experts[i], experts[i])} {means[i]:+.2f}" for i in order)
    top = experts[order[0]]
    own = means[list(experts).index(task)] if task in experts else None
    lines = [f"**Mean weight per expert over the 12 layers:** {ranked}."]
    if top == task:
        lines.append(f"The router gives the most weight to the {TASK_NAMES[task]} expert, the task of this input.")
    elif own is not None:
        lines.append(f"The router gives the most weight to the {TASK_NAMES.get(top, top)} expert, "
                     f"not to the {TASK_NAMES[task]} expert ({own:+.2f}).")
    return "\n\n".join(lines)


def build_app(backend: DemoBackend, runs: dict) -> gr.Blocks:
    run_choices = [(label, name) for name, label in runs.items()]
    default_run = next(iter(runs))
    lam = backend.task_arithmetic_lambda

    def example_choices(task):
        return [(example_title(ex), i) for i, ex in enumerate(EXAMPLES[task])]

    def gold_text(task, example):
        return f"Gold label: **{pretty(example['label'])}** (GLUE validation #{example['idx']})"

    def on_task(task):
        labels = FIELD_LABELS[task]
        first = EXAMPLES[task][0]
        texts = list(first["texts"]) + [""] * (2 - len(first["texts"]))
        return (
            f"**{TASK_NAMES[task]}**: {TASK_QUESTIONS[task]}",
            gr.update(label=labels[0], value=texts[0]),
            gr.update(label=labels[-1], value=texts[1], visible=len(labels) == 2),
            gr.update(choices=example_choices(task), value=0),
            gold_text(task, first),
        )

    def on_example(task, index):
        if index is None:
            return gr.update(), gr.update(), ""
        example = EXAMPLES[task][int(index)]
        texts = list(example["texts"]) + [""] * (2 - len(example["texts"]))
        return texts[0], texts[1], gold_text(task, example)

    def on_predict(task, run, text_a, text_b):
        texts = [text_a, text_b][: len(FIELD_LABELS[task])]
        try:
            result = backend.predict(task, texts, run)
        except ValueError as error:
            raise gr.Error(str(error))
        outputs = [{pretty(name): float(p) for name, p in zip(result.label_names, result.probs[key])}
                   for key in ("expert", "task_arithmetic", "wemoe")]
        figure = routing_figure(result.routing, result.expert_names, task, lam)
        summary = routing_summary(result.routing, result.expert_names, task)
        return (*outputs, figure, summary)

    first = EXAMPLES[DEFAULT_TASK][0]
    with gr.Blocks(title="WEMoE demo") as app:
        gr.Markdown(INTRO)
        with gr.Row():
            with gr.Column(scale=5):
                task = gr.Radio([(TASK_NAMES[t], t) for t in backend.tasks], value=DEFAULT_TASK, label="Task")
                run = gr.Dropdown(run_choices, value=default_run, label="WEMoE model",
                                  info="Switching rebuilds the model (10-20 s).")
                question = gr.Markdown(f"**{TASK_NAMES[DEFAULT_TASK]}**: {TASK_QUESTIONS[DEFAULT_TASK]}")
                example = gr.Dropdown(example_choices(DEFAULT_TASK), value=0, label="Example from GLUE validation")
                text_a = gr.Textbox(first["texts"][0], label=FIELD_LABELS[DEFAULT_TASK][0], lines=2)
                text_b = gr.Textbox("", label="Second text", lines=2, visible=False)
                gold = gr.Markdown(gold_text(DEFAULT_TASK, first))
                predict = gr.Button("Predict", variant="primary")
            with gr.Column(scale=5):
                with gr.Row():
                    expert_out = gr.Label(label="Single-task expert", num_top_classes=2)
                    ta_out = gr.Label(label=f"Task Arithmetic (λ = {lam:g})", num_top_classes=2)
                    wemoe_out = gr.Label(label="WEMoE", num_top_classes=2)
                plot = gr.Plot(label="Routing weights")
                summary = gr.Markdown()

        outputs = [expert_out, ta_out, wemoe_out, plot, summary]
        inputs = [task, run, text_a, text_b]
        task.change(on_task, task, [question, text_a, text_b, example, gold])
        example.input(on_example, [task, example], [text_a, text_b, gold])
        text_a.input(lambda: "", None, gold)
        text_b.input(lambda: "", None, gold)
        predict.click(on_predict, inputs, outputs)
        run.change(on_predict, inputs, outputs)
        app.load(on_predict, inputs, outputs)
    return app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint-dir", default=str(ROOT / "checkpoints"))
    parser.add_argument("--results-dir", default=str(ROOT / "results"))
    parser.add_argument("--device", default=None, help="cuda, mps or cpu (default: the first one available)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true", help="create a public gradio.live link")
    args = parser.parse_args()

    runs = demo_runs(args.checkpoint_dir)
    if not runs:
        sys.exit(f"None of the demo runs has a checkpoint in {Path(args.checkpoint_dir) / 'wemoe'}")
    device = torch.device(args.device) if args.device else default_device()
    print(f"Loading the experts, Task Arithmetic and WEMoE on {device} (about 30 s)...", flush=True)
    backend = DemoBackend.from_checkpoints(args.checkpoint_dir, args.results_dir, DEFAULT_TASKS, device=device)
    warmup = EXAMPLES[DEFAULT_TASK][0]["texts"]
    backend.predict(DEFAULT_TASK, warmup, next(iter(runs)))  # builds the default WEMoE run
    print(f"Runs: {', '.join(runs)}", flush=True)

    app = build_app(backend, runs)
    app.queue(default_concurrency_limit=1)  # one model in memory, one request at a time
    app.launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
