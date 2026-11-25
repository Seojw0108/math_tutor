"""① Qwen3 chat template 에서 enable_thinking=False 로 assistant 토큰만 손실 걸리는지 확인.

검사 내용
  1) TRL assistant_only_loss 지원 여부: 템플릿에 {% generation %} 태그가 있는가 (Qwen3 는 없음 -> 직접 마스킹)
  2) 우리 렌더링(tutor.chat)의 각 선생님 턴 '앞부분'이 공식 템플릿
     apply_chat_template(messages[:i], add_generation_prompt=True, enable_thinking=False) 의 꼬리와 같은가
  3) 대화 1건 전체를 렌더링했을 때 공식 템플릿 출력과 어디가 다른가 (이전 턴 think 블록 유무만 달라야 함)
  4) labels != -100 인 토큰을 디코딩하면 정확히 선생님 말 + <|im_end|> 들인가

  python scripts/check_masking.py --model models/Qwen3-4B [--data data/dialogues.jsonl]
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from transformers import AutoTokenizer  # noqa: E402

from tutor.chat import EMPTY_THINK, IGNORE_INDEX, IM_END, clean_messages, render_prompt, tokenize_for_sft  # noqa: E402
from tutor.io_utils import read_jsonl  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="models/Qwen3-4B")
ap.add_argument("--data", default=None, help="dialogues.jsonl (없으면 tests 의 샘플 대화)")
a = ap.parse_args()

tok = AutoTokenizer.from_pretrained(a.model)
tmpl = tok.chat_template or ""
print("[1] {% generation %} in template:", "{% generation %}" in tmpl,
      "-> TRL assistant_only_loss " + ("사용 가능" if "{% generation %}" in tmpl else "불가, 직접 마스킹(tutor.chat) 사용"))

if a.data:
    dialogues = read_jsonl(a.data)[:20]
else:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
    from test_metrics import SAMPLE
    dialogues = [SAMPLE]

ok_all = True
for d in dialogues:
    msgs = clean_messages(d["messages"])
    # [2] 각 assistant 턴의 생성 프롬프트 꼬리 비교
    for i, m in enumerate(msgs):
        if m["role"] != "assistant":
            continue
        official = tok.apply_chat_template(msgs[:i], tokenize=False, add_generation_prompt=True, enable_thinking=False)
        ours = render_prompt(msgs[:i], thinking=False)
        # 마지막 user 턴 + 생성 머리가 같아야 한다
        tail_o = official[official.rfind("<|im_start|>user"):]
        tail_u = ours[ours.rfind("<|im_start|>user"):]
        if tail_o != tail_u:
            ok_all = False
            print("  MISMATCH tail at turn", i, repr(tail_o[-80:]), repr(tail_u[-80:]))
        # 첫 assistant 턴 전까지는 완전히 같아야 한다
        if i == next(j for j, x in enumerate(msgs) if x["role"] == "assistant") and official != ours:
            ok_all = False
            print("  MISMATCH first prompt")
    # [4] 라벨 검사
    enc = tokenize_for_sft(msgs, tok, max_length=8192)
    lab = [t for t, l in zip(enc["input_ids"], enc["labels"]) if l != IGNORE_INDEX]
    decoded = tok.decode(lab)
    expected = "".join(m["content"].strip() + IM_END for m in msgs if m["role"] == "assistant")
    if decoded != expected:
        ok_all = False
        print("  LABEL MISMATCH\n   got:", repr(decoded[:200]), "\n   exp:", repr(expected[:200]))
    if EMPTY_THINK.strip() in decoded:
        ok_all = False
        print("  think tokens are inside labels!")

print("[2] 각 선생님 턴 생성 프롬프트 = 공식 템플릿(enable_thinking=False):", "OK" if ok_all else "FAIL")
print("[4] 손실 토큰 = 선생님 말 + <|im_end|> 뿐:", "OK" if ok_all else "FAIL")

d = dialogues[0]
msgs = clean_messages(d["messages"])
official_full = tok.apply_chat_template(msgs, tokenize=False, enable_thinking=False)
print("[3] 공식 템플릿은 이전 선생님 턴의 think 블록을 지운다. 우리는 모든 턴에 빈 think 블록을 넣는다 (학습=추론 모양 통일).")
print("    공식 전체 렌더링에 들어 있는 빈 think 블록 수:", official_full.count("<think>"),
      "/ 우리:", render_prompt(msgs[:-1]).count("<think>"))

enc = tokenize_for_sft(msgs, tok)
n_train = sum(1 for l in enc["labels"] if l != IGNORE_INDEX)
print(json.dumps({"tokens": len(enc["input_ids"]), "train_tokens": n_train,
                  "train_ratio": round(n_train / len(enc["input_ids"]), 3)}, ensure_ascii=False))
sys.exit(0 if ok_all else 1)
