"""① Qwen3-4B 받기 -> models/Qwen3-4B (models/ 는 git 제외).

  python scripts/download_model.py                       # Qwen/Qwen3-4B
  python scripts/download_model.py --repo Qwen/Qwen3-0.6B --out models/Qwen3-0.6B   # 빠른 점검용
"""
import argparse

from huggingface_hub import snapshot_download

ap = argparse.ArgumentParser()
ap.add_argument("--repo", default="Qwen/Qwen3-4B")
ap.add_argument("--out", default="models/Qwen3-4B")
a = ap.parse_args()
path = snapshot_download(a.repo, local_dir=a.out, allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model",
                                                                 "merges.txt", "vocab.json", "tokenizer*"])
print("saved to", path)
