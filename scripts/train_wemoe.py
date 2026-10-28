#!/usr/bin/env python
"""Experiment 2: build WEMoE from the experts and train its routers with FSFT.

Examples:
    python scripts/train_wemoe.py --set wemoe.seed=1
    python scripts/train_wemoe.py --set wemoe.k=64 --set wemoe.moe_scope=intermediate
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wemoe_nlp.cli import base_parser, load_run_context  # noqa: E402
from wemoe_nlp.fsft import FSFTConfig, run_dirs, train_wemoe  # noqa: E402
from wemoe_nlp.utils import dataclass_from_dict  # noqa: E402


def main() -> None:
    parser = base_parser(__doc__)
    parser.add_argument("--overwrite", action="store_true", help="rerun even if results exist")
    args = parser.parse_args()
    ctx = load_run_context(args)
    cfg = dataclass_from_dict(FSFTConfig, ctx.cfg.get("wemoe"))
    _, out_dir = run_dirs(ctx.checkpoint_dir, ctx.results_dir, cfg.resolve_run_name())
    if (out_dir / "results.json").exists() and not args.overwrite:
        logging.info("Skipping %s: %s exists (use --overwrite to rerun)", cfg.resolve_run_name(), out_dir)
        return
    train_wemoe(ctx.tasks, ctx.data, cfg, ctx.checkpoint_dir, ctx.results_dir, ctx.device)


if __name__ == "__main__":
    main()
