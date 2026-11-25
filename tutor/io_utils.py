from __future__ import annotations

import csv
import json
import random
from pathlib import Path
from typing import Iterable, Iterator

from .prompts import build_system

RES = Path(__file__).resolve().parent / "resources"


def read_jsonl(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def iter_jsonl(path: str | Path) -> Iterator[dict]:
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> int:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


def load_standards() -> dict[str, dict]:
    """부록의 성취기준 121개. {"[4수01-15]": {grade_band, area, text}}"""
    out = {}
    with open(RES / "standards.tsv", encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            out[row["standard"]] = row
    return out


def select_sft_dialogues(dialogues: list[dict], mode: str = "filtered", fraction: float = 1.0,
                         seed: int = 42) -> list[dict]:
    """SFT 데이터 고르기.
    filtered: split=train & filters.passed=true (본 학습)
    raw:      split=train 전체, passed 무시 (Raw vs Filtered 실험)
    fraction: 데이터 크기 실험 (25%/50%/100%). 같은 seed 면 25% ⊂ 50% ⊂ 100%.
    """
    rows = [d for d in dialogues if d.get("split") == "train"]
    if mode == "filtered":
        rows = [d for d in rows if (d.get("filters") or {}).get("passed")]
    elif mode != "raw":
        raise ValueError(mode)
    rows = sorted(rows, key=lambda d: d["id"])
    random.Random(seed).shuffle(rows)
    k = max(1, int(round(len(rows) * fraction))) if rows else 0
    return rows[:k]


def ensure_system(messages: list[dict], grade_band: str, standard_name: str) -> list[dict]:
    """첫 메시지가 system 이 아니면 고정 템플릿을 붙인다."""
    if messages and messages[0]["role"] == "system":
        return messages
    return [{"role": "system", "content": build_system(grade_band, standard_name)}] + messages
