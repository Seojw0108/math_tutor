"""실험 기록 (공고 요구). 모든 실험은 runs/{날짜}_{이름}/ 에 설정 파일과 로그를 남긴다.

기록 항목: 학습 방법, LR, batch, epoch, optimizer, LoRA rank, trainable params, GPU, 학습 시간, GPU 메모리
+ 왜 그 값을 골랐는지(config 의 `why`).
"""
from __future__ import annotations

import datetime as _dt
import json
import platform
import subprocess
import time
from pathlib import Path

try:
    import yaml  # type: ignore
except ImportError:  # pragma: no cover
    yaml = None


def load_config(path: str | Path, overrides: list[str] | None = None) -> dict:
    """YAML 설정 + CLI 덮어쓰기(--set key=value, 점 표기 가능: lora.r=8)."""
    text = Path(path).read_text(encoding="utf-8")
    cfg = yaml.safe_load(text) if yaml else json.loads(text)
    for ov in overrides or []:
        key, val = ov.split("=", 1)
        try:
            parsed = yaml.safe_load(val) if yaml else json.loads(val)
        except Exception:
            parsed = val
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = parsed
    return cfg


def gpu_info() -> dict:
    info: dict = {"host": platform.node()}
    try:
        import torch
        info["torch"] = torch.__version__
        if torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0)
            info.update({
                "gpu": p.name, "gpu_count": torch.cuda.device_count(),
                "vram_gb": round(p.total_memory / 2**30, 1),
                "bf16": torch.cuda.is_bf16_supported(),
                "cuda": torch.version.cuda,
            })
        else:
            info["gpu"] = "cpu"
    except Exception as e:  # pragma: no cover
        info["gpu_error"] = repr(e)
    return info


def git_rev() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return None


class RunLogger:
    def __init__(self, name: str, root: str | Path = "runs", config: dict | None = None):
        date = _dt.datetime.now().strftime("%Y%m%d")
        self.dir = Path(root) / f"{date}_{name}"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.t0 = time.time()
        self.config = config or {}
        self._dump("config", {**self.config, "_env": gpu_info(), "_git": git_rev(),
                              "_started": _dt.datetime.now().isoformat(timespec="seconds")})
        self.log_path = self.dir / "log.jsonl"

    def _dump(self, stem: str, obj: dict):
        obj = json.loads(json.dumps(obj, ensure_ascii=False, default=str))  # torch 버전 같은 str 하위 타입 정리
        if yaml:
            (self.dir / f"{stem}.yaml").write_text(yaml.safe_dump(obj, allow_unicode=True, sort_keys=False), encoding="utf-8")
        else:
            (self.dir / f"{stem}.json").write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")

    def log(self, **row):
        row = {"t": round(time.time() - self.t0, 1), **row}
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def finish(self, **summary) -> dict:
        out = {"elapsed_sec": round(time.time() - self.t0, 1), **summary}
        try:
            import torch
            if torch.cuda.is_available():
                out["peak_gpu_mem_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
                out["peak_gpu_reserved_gb"] = round(torch.cuda.max_memory_reserved() / 2**30, 2)
        except Exception:
            pass
        (self.dir / "summary.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        return out


def count_trainable(model) -> dict:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return {"trainable_params": trainable, "total_params": total,
            "trainable_pct": round(100 * trainable / max(total, 1), 4)}


def warmup_kwargs(args_cls, ratio: float) -> dict:
    """transformers 4.x: warmup_ratio / 5.x: warmup_steps 에 0~1 실수(비율)."""
    import dataclasses
    names = {f.name for f in dataclasses.fields(args_cls)}
    return {"warmup_ratio": ratio} if "warmup_ratio" in names else {"warmup_steps": ratio}
