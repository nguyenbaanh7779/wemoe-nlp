"""Argument parsing shared by the scripts in ``scripts/``."""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import torch

from .data import DataConfig
from .utils import dataclass_from_dict, get_device, load_config, setup_logging


def base_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", default="configs/default.yaml", help="YAML config (default: %(default)s)")
    parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE",
                        help="override a config value, e.g. --set wemoe.k=64 (repeatable)")
    parser.add_argument("--tasks", nargs="+", help="subset of the configured tasks")
    parser.add_argument("--device", default=None, help="cuda, cpu, ... (default: cuda if available)")
    return parser


@dataclass
class RunContext:
    cfg: dict
    data: DataConfig
    tasks: list
    checkpoint_dir: str
    results_dir: str
    device: torch.device


def load_run_context(args: argparse.Namespace) -> RunContext:
    setup_logging()
    cfg = load_config(args.config, args.overrides)
    paths = cfg.get("paths", {})
    return RunContext(
        cfg=cfg,
        data=dataclass_from_dict(DataConfig, cfg.get("data")),
        tasks=list(args.tasks or cfg["tasks"]),
        checkpoint_dir=paths.get("checkpoint_dir", "checkpoints"),
        results_dir=paths.get("results_dir", "results"),
        device=get_device(args.device),
    )
