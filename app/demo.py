#!/usr/bin/env python
"""Interactive WEMoE demo: one input, three models (expert, Task Arithmetic, WEMoE) and the routing weights.

    pip install -r requirements-demo.txt
    python app/demo.py                      # http://127.0.0.1:7860
    python app/demo.py --share              # also a public link (Colab)
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import gradio as gr  # noqa: E402

from examples import EXAMPLES, example_title  # noqa: E402
from wemoe_nlp.data import DEFAULT_TASKS, TASKS  # noqa: E402
from wemoe_nlp.demo import DEFAULT_RUN, DemoBackend, default_device  # noqa: E402

AUTO = "auto"
TASK_NAMES = {"mrpc": "MRPC", "rte": "RTE", "cola": "CoLA", "sst2": "SST-2", "qnli": "QNLI"}
TASK_QUESTIONS = {  # short form, for the "What it asks" column of the results table
    "mrpc": "Do the two sentences mean the same thing?",
    "rte": "Does sentence 1 imply sentence 2?",
    "cola": "Is the sentence grammatical English?",
    "sst2": "Is the movie review positive or negative?",
    "qnli": "Does the sentence answer the question?",
}
TASK_DESCRIPTIONS = {  # shown under the task selector
    "mrpc": ("**MRPC** (Microsoft Research Paraphrase Corpus): *paraphrase detection*.\n\n"
             "Two sentences from news articles: do they say the same thing?\n\n"
             "Labels: **equivalent** / **not equivalent**."),
    "rte": ("**RTE** (Recognizing Textual Entailment): *textual entailment*.\n\n"
            "Can sentence 2 be inferred from sentence 1?\n\n"
            "Labels: **entailment** / **not entailment**. Only about 2.5k training pairs: the hardest task here."),
    "cola": ("**CoLA** (Corpus of Linguistic Acceptability): *grammatical acceptability*.\n\n"
             "Is the sentence grammatical English? The sentences come from linguistics textbooks.\n\n"
             "Labels: **acceptable** / **unacceptable**. Scored with the Matthews correlation (MCC)."),
    "sst2": ("**SST-2** (Stanford Sentiment Treebank): *sentiment analysis*.\n\n"
             "Is a sentence from a movie review positive or negative? "
             "The text is lower-cased, with spaces before punctuation.\n\n"
             "Labels: **positive** / **negative**."),
    "qnli": ("**QNLI** (Question NLI, built from SQuAD on Wikipedia): *question answering as classification*.\n\n"
             "Does the sentence contain the answer to the question?\n\n"
             "Labels: **entailment** (it answers the question) / **not entailment**."),
}
FIELD_LABELS = {
    AUTO: ("Sentence 1", "Sentence 2 (leave empty for a single sentence)"),
    "mrpc": ("Sentence 1", "Sentence 2"),
    "rte": ("Sentence 1", "Sentence 2"),
    "cola": ("Sentence",),
    "sst2": ("Sentence",),
    "qnli": ("Question", "Sentence"),
}
AUTO_HELP = ("**Auto**: the input is scored for every task that fits it. "
             "One sentence: CoLA and SST-2. Two sentences: MRPC, RTE and QNLI.")
DEFAULT_TASK = "sst2"  # Auto is offered last: its extra rows have no gold label and are hard to read
DEFAULT_EXAMPLE = "sst2:1"  # "it 's slow -- very , very slow .": Task Arithmetic gets it wrong, WEMoE right

# Same diverging blue <-> red scale as docs/report/figures/routing.png: blue adds a task vector, red subtracts it
INK, INK_2, GRID, AXIS = "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7"
SURFACE, HIGHLIGHT = "#fcfcfb", "#eda100"
ADD, SUBTRACT = "#3987e5", "#e34948"
DIVERGING = LinearSegmentedColormap.from_list("blue_gray_red", ["#a32a2a", SUBTRACT, "#f0efec", ADD, "#184f95"])

INTRO = """
# WEMoE: five BERT experts merged into one model

Five BERT-base models were fine-tuned separately on five GLUE tasks and then merged.
For one input, this demo compares three models that serve all five tasks:

- **Single-task expert**: the model fine-tuned on that task only (the upper bound).
- **Task Arithmetic**: one static merge, θ₀ + λ·Σ τₜ, with the same weights for every input.
- **WEMoE** (full MLP): in every layer, a router reads the input and mixes the experts' MLP task vectors with its
  own weights. The routers were trained on 256 labelled examples per task.

The routers see only the text, never the task: the task only picks the classification head at the end.
That is why one input has one set of routing weights, whichever task it is scored for.
"""


def pretty(label: str) -> str:
    return label.replace("_", " ")


def parse_example(value):
    """Dropdown value ``"task:index"`` -> (task, example dict)."""
    task, index = str(value).split(":")
    return task, EXAMPLES[task][int(index)]


def example_choices(task):
    tasks = DEFAULT_TASKS if task == AUTO else (task,)
    return [(example_title(ex, TASK_NAMES[t] if task == AUTO else ""), f"{t}:{i}")
            for t in tasks for i, ex in enumerate(EXAMPLES[t])]


def gold_text(gold):
    if not gold:
        return ""
    return f"Gold label ({TASK_NAMES[gold['task']]}, GLUE validation #{gold['idx']}): **{pretty(gold['label'])}**"


def results_table(result, gold, lam: float) -> str:
    header = (f"| Task | What it asks | Single-task expert | Task Arithmetic (λ = {lam:g}) | WEMoE |\n"
              "|---|---|---|---|---|")
    rows = []
    for task in result.tasks:
        cells = []
        for model in ("expert", "task_arithmetic", "wemoe"):
            label = result.predicted_label(task, model)
            confidence = max(result.probs[task][model])
            mark = ""
            if gold and gold["task"] == task:
                mark = " ✓" if label == gold["label"] else " ✗"
            cells.append(f"{pretty(label)} ({confidence:.2f}){mark}")
        rows.append(f"| **{TASK_NAMES[task]}** | {TASK_QUESTIONS[task]} | " + " | ".join(cells) + " |")
    note = "\n\nNumbers are the probability of the predicted label."
    if gold and gold["task"] in result.tasks:
        note += " ✓ / ✗: agrees / disagrees with the gold label."
    return "\n".join([header] + rows) + note


def routing_bars(routing: np.ndarray, experts, task, lam: float) -> Figure:
    """Mean routing weight per expert over the layers, with Task Arithmetic's constant lambda as reference.

    A plain ``Figure`` (not pyplot), so figures are freed once Gradio has rendered them.
    """
    means = routing.mean(axis=0)
    x = np.arange(len(experts))
    fig = Figure(figsize=(5.4, 3.6), dpi=150, facecolor=SURFACE)
    ax = fig.subplots()
    ax.set_facecolor(SURFACE)
    ax.bar(x, means, width=0.32, color=[ADD if m >= 0 else SUBTRACT for m in means], zorder=3)
    ax.axhline(0, color=AXIS, linewidth=1, zorder=2)
    ax.axhline(lam, color=INK_2, linewidth=1, linestyle=(0, (4, 3)), zorder=2)
    for xi, m in zip(x, means):
        ax.annotate(f"{m:+.2f}", (xi, m), xytext=(0, 3 if m >= 0 else -3), textcoords="offset points",
                    ha="center", va="bottom" if m >= 0 else "top", fontsize=8, color=INK)
    low, high = min(0.0, float(means.min())), max(lam, float(means.max()))
    pad = 0.18 * (high - low) + 0.05
    ax.set_ylim(low - (pad if low < 0 else 0.05), high + pad)
    ax.set_xlim(-0.6, len(experts) - 0.4)
    ax.tick_params(length=0, labelsize=8, colors=INK_2)
    ax.set_xticks(x, [TASK_NAMES.get(e, e) for e in experts], fontsize=8.5)
    for label, expert in zip(ax.get_xticklabels(), experts):  # after tick_params, which would reset the colors
        label.set_color(INK if expert == task else INK_2)
        label.set_fontweight("bold" if expert == task else "normal")
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_ylabel("Mean weight over the 12 layers", fontsize=8.5, color=INK_2)
    xlabel = "Expert task vector" + (" (selected task in bold)" if task in experts else "")
    ax.set_xlabel(xlabel, fontsize=8.5, color=INK_2)
    ax.set_title("WEMoE: how much of each expert this input gets", fontsize=9, color=INK, loc="left", pad=16)
    ax.text(0, 1.02, f"Dashed line: Task Arithmetic, {lam:g} for every expert", transform=ax.transAxes,
            ha="left", va="bottom", fontsize=7.5, color=INK_2)
    fig.tight_layout()
    return fig


def routing_heatmap(routing: np.ndarray, experts, task) -> Figure:
    """Heatmap of the routing weights of one input: layers (rows) x expert task vectors (columns)."""
    layers, num_experts = routing.shape
    limit = max(float(np.abs(routing).max()), 0.5)
    fig = Figure(figsize=(5.4, 5.6), dpi=150, facecolor=SURFACE)
    ax = fig.subplots()
    image = ax.imshow(routing, cmap=DIVERGING, norm=TwoSlopeNorm(vcenter=0.0, vmin=-limit, vmax=limit), aspect="auto")
    for i in range(layers):
        for j in range(num_experts):
            value = routing[i, j]
            color = "white" if abs(value) > 0.6 * limit else INK
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=7.5, color=color)
    xlabel = "Expert task vector"
    if task in experts:
        column = list(experts).index(task)
        ax.add_patch(Rectangle((column - 0.5, -0.5), 1, layers, fill=False, edgecolor=HIGHLIGHT, linewidth=2.5))
        xlabel += f" ({TASK_NAMES[task]} outlined)"
    ax.set_xticks(range(num_experts), [TASK_NAMES.get(e, e) for e in experts], fontsize=8.5, color=INK_2)
    ax.set_yticks(range(layers), [str(i + 1) for i in range(layers)], fontsize=8, color=INK_2)
    ax.set_xlabel(xlabel, fontsize=8.5, color=INK_2)
    ax.set_ylabel("Layer", fontsize=8.5, color=INK_2)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title("WEMoE routing weights for this input", fontsize=9, color=INK, loc="left")
    bar = fig.colorbar(image, ax=ax, fraction=0.05, pad=0.03)
    bar.outline.set_visible(False)
    bar.ax.tick_params(labelsize=7, colors=INK_2)
    fig.tight_layout()
    return fig


def routing_summary(routing: np.ndarray, experts, task, lam: float) -> str:
    means = routing.mean(axis=0)
    order = np.argsort(-means)
    ranked = ", ".join(f"{TASK_NAMES.get(experts[i], experts[i])} {means[i]:+.2f}" for i in order)
    lines = [f"**Mean weight per expert over the 12 layers:** {ranked}. "
             f"Task Arithmetic uses {lam:g} for every expert in every layer."]
    if task in experts:
        top = experts[order[0]]
        if top == task:
            lines.append(f"The router gives the most weight to the {TASK_NAMES[task]} expert.")
        else:
            own = means[list(experts).index(task)]
            lines.append(f"The router gives the most weight to the {TASK_NAMES.get(top, top)} expert, "
                         f"not to the {TASK_NAMES[task]} expert ({own:+.2f}).")
    return "\n\n".join(lines)


def build_app(backend: DemoBackend) -> gr.Blocks:
    lam = backend.task_arithmetic_lambda

    def fill(task, example_value):
        """Texts and gold label of an example, for the field layout of ``task``."""
        example_task, example = parse_example(example_value)
        texts = list(example["texts"]) + [""] * (2 - len(example["texts"]))
        gold = {"task": example_task, "label": example["label"], "idx": example["idx"]}
        return texts[0], texts[1], gold, gold_text(gold)

    def on_task(task, example_value):
        labels = FIELD_LABELS[task]
        if task != AUTO and (not example_value or parse_example(example_value)[0] != task):
            example_value = f"{task}:0"
        question = AUTO_HELP if task == AUTO else TASK_DESCRIPTIONS[task]
        text_a, text_b, gold, gold_md = fill(task, example_value)
        return (question, gr.update(label=labels[0], value=text_a),
                gr.update(label=labels[-1], value=text_b, visible=len(labels) == 2),
                gr.update(choices=example_choices(task), value=example_value), gold, gold_md)

    def on_example(task, example_value):
        if not example_value:
            return gr.update(), gr.update(), gr.update(), gr.update()
        return fill(task, example_value)

    def on_predict(task, text_a, text_b, gold):
        if task == AUTO:
            texts, tasks = [text_a] + ([text_b] if text_b.strip() else []), None
        else:
            texts, tasks = [text_a, text_b][: len(TASKS[task].text_fields)], [task]
        try:
            result = backend.predict(texts, tasks)
        except ValueError as error:
            raise gr.Error(str(error))
        selected = None if task == AUTO else task
        routing, experts = result.routing, result.expert_names
        return (results_table(result, gold, lam), routing_bars(routing, experts, selected, lam),
                routing_heatmap(routing, experts, selected), routing_summary(routing, experts, selected, lam))

    default_a, default_b, default_gold, default_gold_md = fill(DEFAULT_TASK, DEFAULT_EXAMPLE)
    default_labels = FIELD_LABELS[DEFAULT_TASK]
    with gr.Blocks(title="WEMoE demo") as app:
        gr.Markdown(INTRO)
        gold = gr.State(default_gold)
        with gr.Row():
            with gr.Column(scale=4):
                task = gr.Radio([(TASK_NAMES[t], t) for t in backend.tasks] + [("Auto", AUTO)], value=DEFAULT_TASK,
                                label="Task")
                question = gr.Markdown(TASK_DESCRIPTIONS[DEFAULT_TASK])
                example = gr.Dropdown(example_choices(DEFAULT_TASK), value=DEFAULT_EXAMPLE,
                                      label="Example from GLUE validation")
                text_a = gr.Textbox(default_a, label=default_labels[0], lines=2)
                text_b = gr.Textbox(default_b, label=default_labels[-1], lines=2, visible=len(default_labels) == 2)
                gold_md = gr.Markdown(default_gold_md)
                predict = gr.Button("Predict", variant="primary")
            with gr.Column(scale=6):
                table = gr.Markdown()
                with gr.Tabs():  # PNG: the default WebP smears the heatmap cells
                    with gr.Tab("Mean over layers"):
                        bars = gr.Plot(label="Mean routing weight per expert", format="png")
                    with gr.Tab("Per layer (heatmap)"):
                        heatmap = gr.Plot(label="Routing weight per layer and expert", format="png")
                summary = gr.Markdown()

        inputs, outputs = [task, text_a, text_b, gold], [table, bars, heatmap, summary]
        task.change(on_task, [task, example], [question, text_a, text_b, example, gold, gold_md]).then(
            on_predict, inputs, outputs)
        example.input(on_example, [task, example], [text_a, text_b, gold, gold_md]).then(on_predict, inputs, outputs)
        for box in (text_a, text_b):
            box.input(lambda: (None, ""), None, [gold, gold_md])  # edited text: the gold label no longer applies
        predict.click(on_predict, inputs, outputs)
        app.load(on_predict, inputs, outputs)
    return app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint-dir", default=str(ROOT / "checkpoints"))
    parser.add_argument("--results-dir", default=str(ROOT / "results"))
    parser.add_argument("--run", default=DEFAULT_RUN, help="WEMoE run in checkpoints/wemoe/ (default: %(default)s)")
    parser.add_argument("--device", default=None, help="cuda, mps or cpu (default: the first one available)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true", help="create a public gradio.live link")
    args = parser.parse_args()

    device = torch.device(args.device) if args.device else default_device()
    print(f"Loading the experts, Task Arithmetic and WEMoE ({args.run}) on {device}...", flush=True)
    backend = DemoBackend.from_checkpoints(args.checkpoint_dir, args.results_dir, args.run, DEFAULT_TASKS,
                                           device=device)
    backend.predict(list(parse_example(DEFAULT_EXAMPLE)[1]["texts"]), [DEFAULT_TASK])  # warm-up
    app = build_app(backend)
    app.queue(default_concurrency_limit=1)  # one request at a time
    app.launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
