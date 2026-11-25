"""기존 수학 실력 유지 확인 (기획서: "수학 지식이 아니라 가르치는 법·말투를 가르친다, 기존 수학 실력은 유지").

선생님 프롬프트 없이 계산만 묻고 정답률을 Base / SFT / SFT+DPO 끼리 비교한다.
문제: 확정 오개념 8종(뺄셈 받아내림, 나머지 나눗셈, 분수 덧셈)을 새 숫자로 + 곱셈·혼합 계산·소수. 오개념이 갈리는 숫자만.

  python scripts/math_retention.py --models base,sft,sft_dpo --sft_adapter ... --dpo_adapter ... --n 200
  -> runs/{날짜}_math_retention/results.jsonl, summary.json (종류별 정답률)
"""
import argparse
import random
import re
import sys
from collections import defaultdict
from fractions import Fraction
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tutor.generation import load_teacher  # noqa: E402
from tutor.io_utils import write_jsonl  # noqa: E402
from tutor.runlog import RunLogger  # noqa: E402
from tutor.students import RULES, solve  # noqa: E402

ASK = "다음을 계산하세요. 풀이 없이 답만 쓰세요. 나머지가 있으면 '몫…나머지' 꼴로 쓰세요.\n{q} ="


def make_items(n: int, seed: int = 0) -> list[dict]:
    rng = random.Random(seed)
    items = []
    kinds = list(RULES) + ["곱셈", "혼합계산", "소수덧셈"]
    while len(items) < n:
        k = kinds[len(items) % len(kinds)]
        if k in RULES:
            kind, rule = RULES[k]
            if kind == "sub":
                a = rng.randint(100, 999) if "0에서" not in k else rng.choice([301, 402, 503, 604, 705, 806])
                b = rng.randint(11, a - 1)
                prob, q = (a, b), f"{a} − {b}"
            elif kind == "div":
                b = rng.randint(3, 9)
                a = b * rng.randint(2, 12) + rng.randint(1, b - 1)
                prob, q = (a, b), f"{a} ÷ {b}"
            else:
                if "분모도더함" in k:
                    d = rng.randint(5, 12)
                    x, y = rng.randint(1, d - 1), rng.randint(1, d - 1)
                    if x + y >= d:
                        continue
                    prob = ((x, d), (y, d))
                else:
                    b, d = rng.sample([2, 3, 4, 5, 6, 8], 2)
                    prob = ((1, b), (1, d))
                (x, b), (y, d) = prob
                q = f"{x}/{b} + {y}/{d}"
            ans = solve(kind, prob)
            if ans == solve(kind, prob, rule):  # 오개념과 답이 같은 문제는 뺀다
                continue
        elif k == "곱셈":
            a, b = rng.randint(12, 99), rng.randint(3, 19)
            q, ans = f"{a} × {b}", str(a * b)
        elif k == "혼합계산":
            a, b, c = rng.randint(2, 20), rng.randint(2, 9), rng.randint(2, 9)
            q, ans = f"{a} + {b} × {c}", str(a + b * c)
        else:
            a, b = rng.randint(1, 99) / 100, rng.randint(1, 99) / 10
            q, ans = f"{a} + {b}", f"{round(a + b, 2):g}"
        items.append({"kind": k, "question": q, "answer": ans})
    return items


def parse_answer(text: str) -> str | None:
    t = text.replace(",", "").replace(" ", "")
    m = re.search(r"(\d+)(?:…|\.\.\.|나머지)(\d+)", t)
    if m:
        return f"{m.group(1)}…{m.group(2)}"
    m = re.search(r"-?\d+(?:\.\d+)?(?:/\d+)?", t)
    return m.group(0) if m else None


def same(pred: str | None, ans: str) -> bool:
    if pred is None:
        return False
    if "…" in ans or "…" in pred:
        return pred == ans
    try:
        return Fraction(pred) == Fraction(ans)
    except (ValueError, ZeroDivisionError):
        return pred == ans


ap = argparse.ArgumentParser()
ap.add_argument("--models", default="base,sft,sft_dpo")
ap.add_argument("--model", default="models/Qwen3-4B")
ap.add_argument("--sft_adapter")
ap.add_argument("--dpo_adapter")
ap.add_argument("--n", type=int, default=200)
ap.add_argument("--batch_size", type=int, default=32)
a = ap.parse_args()

items = make_items(a.n)
log = RunLogger("math_retention", config=vars(a) | {"why": "LoRA 학습 뒤에도 계산 실력이 그대로인지 (기획서 모델 원칙)"})
rows, summary = [], {}
for name in a.models.split(","):
    t = load_teacher(name, a.model, sft_adapter=a.sft_adapter, dpo_adapter=a.dpo_adapter, batch_size=a.batch_size,
                     sampling={"temperature": 0.0, "max_new_tokens": 32})
    outs = t.generate([[{"role": "user", "content": ASK.format(q=it["question"])}] for it in items])
    acc = defaultdict(list)
    for it, o in zip(items, outs):
        pred = parse_answer(o.text)
        ok = same(pred, it["answer"])
        acc[it["kind"]].append(ok)
        rows.append({"model": name, **it, "response": o.text, "pred": pred, "ok": ok})
    summary[name] = {"overall": round(sum(sum(v) for v in acc.values()) / len(items), 4),
                     **{k: round(sum(v) / len(v), 3) for k, v in acc.items()}}
    print(name, summary[name]["overall"])
    del t
write_jsonl(log.dir / "results.jsonl", rows)
log.finish(accuracy=summary)
print(summary)
