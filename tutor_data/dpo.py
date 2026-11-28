"""dpo.jsonl (규칙 쌍) 생성.

PDF 규칙
- 필터 통과 대화(filters.passed == true)에서만 만든다. chosen = 원래 대화의 다음 선생님 말.
- prompt = messages 를 그 시점(아이 턴)까지 자른 것, role·content 만.
- cut_after = 자른 지점 직전 아이 턴의 response_type. 주로 오개념오답 / 모름 / 답요구 / 유행어 / 비속어.
- id = {source_id}_d{n}. 같은 대화에서 여러 쌍 가능.
- rejected_type: 바로정답 / 어려운용어 / 계산오류 / 질문없음 / 오개념무시 / 말투위반 / 안전위반
  (SFT하위 는 B의 SFT 뒤에 추가되므로 여기서 만들지 않는다.)

    python -m tutor_data.dpo --dialogues data/dialogues.jsonl --out data/dpo.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
import sys

from . import spec
from .misconceptions import correct_answer, load_misconceptions, wrong_answer
from .problems import op_of
from .student import parse_problem

# cut_after 별로 그 자리에서 나쁜 선생님 응답이 될 수 있는 유형
REJECT_FOR = {
    "오개념오답": ["바로정답", "오개념무시", "어려운용어", "계산오류", "질문없음", "말투위반"],
    "모름": ["바로정답", "어려운용어", "질문없음", "말투위반"],
    "답요구": ["바로정답", "말투위반", "질문없음"],
    "유행어": ["안전위반", "말투위반"],
    "비속어": ["안전위반", "말투위반"],
}

JARGON = {
    "[4수01-03]": "피감수의 해당 자릿값이 감수보다 작으면 상위 자릿값에서 받아내림을 수행하고, 받아내림한 자릿값은 1 감소시켜야 합니다.",
    "[4수01-06]": "피제수를 제수로 나눈 몫과 나머지를 구할 때 나머지는 반드시 제수보다 작아야 하며, 몫은 제수의 배수 개수를 의미합니다.",
    "[4수01-15]": "동분모 분수의 덧셈에서는 분모를 보존하고 분자의 합만 계산하는 것이 원칙입니다.",
    "[6수01-08]": "이분모 분수의 덧셈은 최소공배수로 통분한 뒤 분자의 합을 구해야 합니다.",
}
RUDE = ["이것도 몰라? 다시 해.", "또 틀렸네. 집중 좀 해.", "그런 것도 못 하면 어떡해. 빨리 다시 풀어."]
SAFETY_BAD = {
    "유행어": ["ㅋㅋ 선생님도 그 말 완전 좋아해! 우리 수업 말고 유행어 얘기 더 하자 ㅋㅋ"],
    "비속어": ["맞아, 이거 진짜 짜증나지. 선생님도 짜증나."],
}


def _current_problem(op: str, prompt: list[dict], fallback: str):
    """prompt 안 마지막으로 낸 계산 문제."""
    for m in reversed(prompt):
        if m["role"] == "assistant":
            p = parse_problem(op, m["content"])
            if p is not None:
                return p
    return parse_problem(op, fallback)


def _miscalc(op: str, ans: str) -> str:
    if op == "sub":
        return str(int(ans) + 10)
    if op == "div":
        q, r = ans.split("…")
        return f"{int(q) + 1}…{r}"
    n, d = ans.split("/")
    return f"{int(n) + 1}/{d}"


def _has_batchim(ans: str) -> bool:
    """숫자 답을 소리 내어 읽을 때 끝 글자에 받침이 있는가 (340 → 사십, 4/9 → 구분의 사)."""
    last = ans.replace("…", " ").replace("/", " ").split()[-1]
    if last.endswith("0"):
        return True  # 십·백·천
    return last[-1] in "13678"  # 일 삼 육 칠 팔


def _ieyo(ans: str) -> str:
    return ans + ("이에요" if _has_batchim(ans) else "예요")


def _eunneun(ans: str) -> str:
    return ans + ("은" if _has_batchim(ans) else "는")


def _expr(op, ops) -> str:
    from .problems import expr
    return expr(op, ops)


def make_rejected(rtype: str, d: dict, prompt: list[dict], cut_after: str, rng: random.Random) -> str:
    mid = d["misconception_id"]
    op = op_of(mid)
    ops = _current_problem(op, prompt, d["problem"]["question"])
    q = _expr(op, ops)
    ans = correct_answer(op, ops)
    steps = d["problem"]["steps"]
    if rtype == "바로정답":
        head = "틀렸어요. " if cut_after == "오개념오답" else ""
        return f"{head}정답은 {_ieyo(ans)}. {steps[-1]}."
    if rtype == "어려운용어":
        return JARGON[d["standard"]] + " 다시 계산해 보세요."
    if rtype == "계산오류":
        last = q.split()[-1]
        return f"같이 풀어 봐요. {_eunneun(q) if '/' not in last else q + ('은' if _has_batchim(last.split('/')[0]) else '는')} {_ieyo(_miscalc(op, ans))}. 이제 알겠죠?"
    if rtype == "질문없음":
        return " ".join(s + "." for s in steps)
    if rtype == "오개념무시":
        return f"맞아요, {wrong_answer(mid, ops)}! 아주 잘했어요. 다음으로 넘어가요."
    if rtype == "말투위반":
        return rng.choice(RUDE)
    if rtype == "안전위반":
        return rng.choice(SAFETY_BAD[cut_after])
    raise ValueError(rtype)


def build_pairs(dialogues: list[dict], seed: int = 0) -> list[dict]:
    rng = random.Random(seed)
    counts = {t: 0 for t in spec.REJECTED_TYPES}
    out = []
    for d in dialogues:
        if d["filters"].get("passed") is not True:
            continue  # chosen 은 필터 통과 대화에서만
        msgs = d["messages"]
        n = 0
        for i, m in enumerate(msgs):
            if m["role"] != "user" or m.get("response_type") not in REJECT_FOR:
                continue
            if i + 1 >= len(msgs) or msgs[i + 1]["role"] != "assistant":
                continue
            cut = m["response_type"]
            prompt = [{"role": x["role"], "content": x["content"]} for x in msgs[: i + 1]]
            # 가장 적게 나온 유형부터 써서 rejected_type 이 고르게 섞이게 한다
            cands = sorted(REJECT_FOR[cut], key=lambda t: (counts[t], rng.random()))
            rtype = cands[0]
            rejected = make_rejected(rtype, d, prompt, cut, rng)
            chosen = msgs[i + 1]["content"]
            if rejected == chosen:
                continue
            counts[rtype] += 1
            n += 1
            out.append({
                "id": f"{d['id']}_d{n}",
                "source_id": d["id"],
                "cut_after": cut,
                "prompt": prompt,
                "chosen": chosen,
                "rejected": rejected,
                "rejected_type": rtype,
            })
    return out


def main(argv=None):
    if sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--dialogues", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    ds = [json.loads(l) for l in open(a.dialogues, encoding="utf-8") if l.strip()]
    passed = sum(d["filters"].get("passed") is True for d in ds)
    if passed == 0:
        sys.exit("filters.passed=true 인 대화가 없습니다. B의 metrics.py 로 filters 를 채운 뒤 실행하세요.")
    pairs = build_pairs(ds, a.seed)
    with open(a.out, "w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"필터 통과 대화 {passed}개 → DPO 쌍 {len(pairs)}개 → {a.out}")


if __name__ == "__main__":
    main()
