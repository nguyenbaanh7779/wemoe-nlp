"""Repository checks: no DVC / HF `evaluate` dependency and no leaked GitHub tokens."""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEXT_SUFFIXES = {".py", ".yaml", ".yml", ".ipynb", ".md", ".toml", ".txt", ".cfg"}


def _repo_text_files():
    for folder in ("src", "scripts", "tests", "configs", "notebooks", "app"):
        base = ROOT / folder
        if base.exists():
            yield from (p for p in base.rglob("*") if p.is_file() and p.suffix in TEXT_SUFFIXES)
    for name in ("README.md", "requirements.txt", "pyproject.toml"):
        if (ROOT / name).exists():
            yield ROOT / name


def test_no_dvc_or_hf_evaluate_imports():
    pattern = re.compile(r"^\s*(?:import|from)\s+(?:dvc|evaluate)\b", re.MULTILINE)
    offenders = [str(p) for p in _repo_text_files() if p.suffix == ".py" and pattern.search(p.read_text("utf-8"))]
    assert not offenders


def test_no_github_tokens():
    token = re.compile(r"gh[pousr]_[A-Za-z0-9]{36}|github_pat_[A-Za-z0-9_]{22,}")
    offenders = [str(p) for p in _repo_text_files() if token.search(p.read_text("utf-8", errors="ignore"))]
    assert not offenders
