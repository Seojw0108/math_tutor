"""⑨ LoRA adapter 를 Hugging Face 에 올리기 (기본 비공개). HF_TOKEN 환경 변수 또는 `huggingface-cli login` 필요.

  python scripts/push_adapter.py --adapter runs/20261205_dpo/adapter --repo_id <계정>/math-tutor-qwen3-4b-lora \
      --summary runs/20261205_dpo/summary.json [--public]

SFT+DPO 는 어댑터가 2개(SFT -> DPO 순서)이므로 둘 다 올리거나, merge_adapters.py 로 SFT 를 병합한 뒤
DPO 어댑터만 올리고 README 에 순서를 적는다. 여기서는 폴더 하나를 올리고 README(모델 카드)를 만든다.
"""
import argparse
import json
from pathlib import Path

from huggingface_hub import HfApi

ap = argparse.ArgumentParser()
ap.add_argument("--adapter", required=True)
ap.add_argument("--repo_id", required=True)
ap.add_argument("--summary", default=None, help="runs/.../summary.json (학습 기록을 모델 카드에)")
ap.add_argument("--base", default="Qwen/Qwen3-4B")
ap.add_argument("--public", action="store_true")
a = ap.parse_args()

summary = json.loads(Path(a.summary).read_text(encoding="utf-8")) if a.summary else {}
card = f"""---
base_model: {a.base}
library_name: peft
language: [ko]
tags: [lora, education, math, tutoring, korean]
---
# 초등 수학 선생님 LoRA ({a.base})

질문하며 가르치는 초등 수학 선생님. 아이의 오답 뒤에 숨은 오개념을 질문으로 짚고, 바로 정답을 말하지 않는다.

## 사용
system 프롬프트(고정): `너는 초등학교 선생님이야. 학년군: {{grade_band}}. 성취기준: {{standard_name}}. 쉬운 말로, 한 번에 하나씩 질문하며 가르쳐.`
첫 user 턴은 `[수업 시작]`. enable_thinking=False.

## 학습 기록
```json
{json.dumps({k: v for k, v in summary.items() if k != 'why'}, ensure_ascii=False, indent=2)}
```
{summary.get('why', '')}
"""
(Path(a.adapter) / "README.md").write_text(card, encoding="utf-8")
api = HfApi()
api.create_repo(a.repo_id, private=not a.public, exist_ok=True)
api.upload_folder(folder_path=a.adapter, repo_id=a.repo_id, commit_message="upload LoRA adapter")
print("uploaded ->", f"https://huggingface.co/{a.repo_id}", "(private)" if not a.public else "(public)")
