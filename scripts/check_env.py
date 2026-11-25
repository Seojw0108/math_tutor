"""① 학습 GPU 확인 (엘리스): VRAM, bf16 지원, 패키지 버전, Qwen3-4B 메모리 어림.

  python scripts/check_env.py
"""
import importlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tutor.runlog import gpu_info  # noqa: E402

info = gpu_info()
for pkg in ["transformers", "trl", "peft", "datasets", "accelerate", "bitsandbytes", "kiwipiepy", "vllm", "gradio"]:
    try:
        info[pkg] = importlib.import_module(pkg).__version__
    except Exception:
        info[pkg] = None

# Qwen3-4B (약 4.0B 파라미터) 메모리 어림 (활성값 제외, 배치·길이에 따라 더 든다)
P = 4.02e9
info["estimate_gb"] = {
    "bf16_weights": round(P * 2 / 2**30, 1),
    "lora_bf16_train(weights+lora+optim, act 제외)": round(P * 2 / 2**30 + 1.5, 1),
    "qlora_4bit_train(weights+lora+optim, act 제외)": round(P * 0.55 / 2**30 + 1.5, 1),
    "full_ft_adamw(참고)": round(P * 16 / 2**30, 1),
}
vram = info.get("vram_gb")
if vram:
    info["verdict"] = ("LoRA bf16 가능 (max_len 4096, batch 1~2 + grad ckpt)" if vram >= 22 else
                       "LoRA bf16 빠듯 -> max_len 2048 또는 QLoRA" if vram >= 14 else "QLoRA 권장")
    if not info.get("bf16"):
        info["verdict"] += " | bf16 미지원 GPU: fp16 + loss scaling 으로 바꿔야 함"
print(json.dumps(info, ensure_ascii=False, indent=2))
