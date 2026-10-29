#!/usr/bin/env python
"""Experiment 4: stored parameters, checkpoint size and (optionally) latency of
5 single-task models vs Task Arithmetic vs WEMoE (full MLP and intermediate-only)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wemoe_nlp.cli import base_parser, load_run_context  # noqa: E402
from wemoe_nlp.efficiency import efficiency_markdown, efficiency_report  # noqa: E402
from wemoe_nlp.experts import load_experts, load_pretrained  # noqa: E402
from wemoe_nlp.utils import save_json  # noqa: E402


def main() -> None:
    parser = base_parser(__doc__)
    parser.add_argument("--latency", action="store_true", help="also time forward passes")
    parser.add_argument("--trained-experts", action="store_true",
                        help="load the trained experts (counts are the same; only needed for real weights)")
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 32])
    args = parser.parse_args()
    ctx = load_run_context(args)

    pretrained = load_pretrained(ctx.data.model_name)
    experts = load_experts(ctx.checkpoint_dir, ctx.tasks) if args.trained_experts else None
    amp = bool(ctx.cfg.get("wemoe", {}).get("fp16", True)) and ctx.device.type == "cuda"
    report = efficiency_report(pretrained, ctx.tasks, experts, latency=args.latency, device=ctx.device,
                               seq_len=args.seq_len, batch_sizes=args.batch_sizes, amp=amp)
    out = Path(ctx.results_dir) / "efficiency.json"
    save_json(report, out)

    print("\n".join(efficiency_markdown(report)))
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
