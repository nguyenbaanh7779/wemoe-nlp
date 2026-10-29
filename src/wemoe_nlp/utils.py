"""Shared helpers: logging, seeding, devices, mixed precision, config files and JSON I/O."""

from __future__ import annotations

import contextlib
import dataclasses
import json
import logging
import random
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import torch
import yaml

logger = logging.getLogger("wemoe_nlp")


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(name: str | None = None) -> torch.device:
    if name:
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def autocast(device: torch.device, enabled: bool):
    """fp16 autocast on CUDA, a no-op elsewhere."""
    if enabled and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return contextlib.nullcontext()


def make_grad_scaler(enabled: bool):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):  # torch < 2.3
        return torch.cuda.amp.GradScaler(enabled=enabled)


# ---------------------------------------------------------------------------
# Config files
# ---------------------------------------------------------------------------


def parse_value(raw: str) -> Any:
    """Parse a CLI override value with YAML rules, also accepting ``2e-5`` style floats."""
    value = yaml.safe_load(raw)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return value
    return value


def set_by_path(cfg: dict, dotted_key: str, value: Any) -> None:
    node = cfg
    *parents, leaf = dotted_key.split(".")
    for part in parents:
        node = node.setdefault(part, {})
    node[leaf] = value


def load_config(path: str | Path, overrides: Iterable[str] = ()) -> dict:
    """Load a YAML config and apply ``section.key=value`` overrides."""
    with open(path, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}
    for item in overrides:
        key, sep, raw = item.partition("=")
        if not sep:
            raise ValueError(f"Override must look like key=value, got {item!r}")
        set_by_path(cfg, key.strip(), parse_value(raw))
    return cfg


def dataclass_from_dict(cls, values: Mapping[str, Any] | None):
    """Build a dataclass from a config section, rejecting unknown keys (typos)."""
    values = dict(values or {})
    known = {f.name for f in dataclasses.fields(cls)}
    unknown = sorted(set(values) - known)
    if unknown:
        raise KeyError(f"Unknown {cls.__name__} keys: {unknown}")
    return cls(**values)


# ---------------------------------------------------------------------------
# JSON I/O
# ---------------------------------------------------------------------------


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, torch.Tensor):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    if dataclasses.is_dataclass(obj):
        return dataclasses.asdict(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def save_json(obj: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, ensure_ascii=False, default=_json_default)


def load_json(path: str | Path) -> Any:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def write_jsonl(rows: Iterable[Mapping[str, Any]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, default=_json_default) + "\n")


def library_versions() -> dict[str, str]:
    import datasets
    import sklearn
    import transformers

    return {
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "datasets": datasets.__version__,
        "scikit-learn": sklearn.__version__,
    }


# ---------------------------------------------------------------------------
# Parameter accounting
# ---------------------------------------------------------------------------


def stored_numel(module: torch.nn.Module) -> int:
    """Number of floating-point values a checkpoint of ``module`` stores (params + buffers)."""
    return sum(t.numel() for t in module.state_dict().values() if torch.is_floating_point(t))


def stored_bytes(module: torch.nn.Module) -> int:
    return sum(
        t.numel() * t.element_size() for t in module.state_dict().values() if torch.is_floating_point(t)
    )


def trainable_numel(module: torch.nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)
