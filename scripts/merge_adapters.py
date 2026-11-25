"""어댑터를 base 에 병합해 하나의 모델 폴더로 저장 (vLLM 평가·데모 배포용).

  python scripts/merge_adapters.py --adapters runs/.../sft/adapter runs/.../dpo/adapter --out models/tutor-sft-dpo
"""
import argparse

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="models/Qwen3-4B")
ap.add_argument("--adapters", nargs="+", required=True, help="붙일 순서대로 (SFT 다음 DPO)")
ap.add_argument("--out", required=True)
a = ap.parse_args()
model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16)
for ad in a.adapters:
    model = PeftModel.from_pretrained(model, ad).merge_and_unload()
model.save_pretrained(a.out, safe_serialization=True)
AutoTokenizer.from_pretrained(a.model).save_pretrained(a.out)
print("merged ->", a.out)
