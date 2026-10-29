"""Helpers shared by the result tables and figures (scripts/summarize_results.py, scripts/plot_results.py)."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Optional, Sequence

from .metrics import majority_metrics


def majority_validation(results_dir, tasks: Sequence[str], baselines: Optional[dict]) -> Optional[dict]:
    """Majority-class metrics on the validation split.

    Read from baselines.json; older files lack them, so fall back to the labels stored in any WEMoE
    prediction file.
    """
    stored = (baselines or {}).get("majority", {}).get("validation")
    if stored:
        return stored
    for path in sorted((Path(results_dir) / "wemoe").glob("*/predictions_validation.jsonl")):
        labels = defaultdict(list)
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                record = json.loads(line)
                labels[record["task"]].append(record["label"])
        if all(t in labels for t in tasks):
            return {t: majority_metrics(t, labels[t]) for t in tasks}
    return None
