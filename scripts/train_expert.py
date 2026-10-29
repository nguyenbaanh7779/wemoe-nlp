#!/usr/bin/env python
"""Experiment 1: fine-tune one BERT expert per GLUE task (best epoch on the dev split)."""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wemoe_nlp.cli import base_parser, load_run_context  # noqa: E402
from wemoe_nlp.experts import ExpertTrainConfig, expert_dir, expert_results_path, train_expert  # noqa: E402
from wemoe_nlp.utils import dataclass_from_dict  # noqa: E402


def main() -> None:
    parser = base_parser(__doc__)
    parser.add_argument("--overwrite", action="store_true", help="retrain experts that already exist")
    args = parser.parse_args()
    ctx = load_run_context(args)
    train_cfg = dataclass_from_dict(ExpertTrainConfig, ctx.cfg.get("expert"))
    for task in ctx.tasks:
        done = (expert_dir(ctx.checkpoint_dir, task) / "config.json").exists() and \
            expert_results_path(ctx.results_dir, task).exists()
        if done and not args.overwrite:
            logging.info("Skipping %s: already trained (use --overwrite to retrain)", task)
            continue
        train_expert(task, ctx.data, train_cfg, ctx.checkpoint_dir, ctx.results_dir, ctx.device)


if __name__ == "__main__":
    main()
