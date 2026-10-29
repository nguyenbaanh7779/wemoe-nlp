#!/usr/bin/env python
"""Evaluate single-task experts, Simple Averaging and Task Arithmetic (lambda chosen on dev)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wemoe_nlp.baselines import BaselineConfig, run_baselines  # noqa: E402
from wemoe_nlp.cli import base_parser, load_run_context  # noqa: E402
from wemoe_nlp.utils import dataclass_from_dict  # noqa: E402


def main() -> None:
    args = base_parser(__doc__).parse_args()
    ctx = load_run_context(args)
    cfg = dataclass_from_dict(BaselineConfig, ctx.cfg.get("baselines"))
    run_baselines(ctx.tasks, ctx.data, cfg, ctx.checkpoint_dir, ctx.results_dir, ctx.device)


if __name__ == "__main__":
    main()
