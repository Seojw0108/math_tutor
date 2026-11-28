"""문제 생성기.

PDF 규칙
- 진단용 문제는 오개념마다 오답이 갈리는 숫자로만 만든다
  (같은 성취기준의 다른 오개념과 오답이 같거나, 오답이 정답과 같으면 버린다).
- split: test 는 학습에 안 쓴 숫자 범위·template_id·context·유행어만 쓴다.
"""
from __future__ import annotations

import random
from math import gcd

from .misconceptions import RULES, correct_answer, same_value, wrong_answer

# ---------------------------------------------------------------- split 별 재료
# 숫자 범위: train / test 가 겹치지 않는다.
NUMBER_RANGES = {
    "sub": {"train": (100, 499), "test": (500, 999)},       # 빼어지는 수 (세 자리 수)
    "div": {"train": (10, 49), "test": (50, 99)},           # 나누어지는 수 (나누는 수는 한 자리)
    "frac_same": {"train": (3, 8), "test": (9, 12)},        # 분모 (분모가 같은 분수)
    "frac_diff": {"train": (2, 6), "test": (7, 12)},        # 분모 (분모가 다른 분수)
}

# context(비유 소재): train / test 가 겹치지 않는다.
CONTEXTS = {
    "sub": {"train": ["사탕", "스티커", "구슬", "색종이"], "test": ["딱지", "우표"]},
    "div": {"train": ["사탕", "연필", "쿠키", "풍선"], "test": ["귤", "구슬"]},
    "frac_add": {"train": ["피자", "케이크", "색종이", "초콜릿"], "test": ["수박", "김밥"]},
}

# template_id(문장제 틀): train / test 가 겹치지 않는다. {c}=context
TEMPLATES = {
    "sub": {
        "train": {
            "T01": "{c}이(가) {a}개 있었는데 {b}개를 친구에게 주었어요. 남은 {c}은(는) 몇 개일까요?",
            "T02": "{c} {a}개 중에서 {b}개를 썼어요. 몇 개가 남았을까요?",
            "T03": "{c}이(가) {a}개, 동생은 {b}개 있어요. 누가 몇 개 더 많을까요?",
        },
        "test": {
            "T04": "{c}을(를) {a}개 모았는데 {b}개를 잃어버렸어요. 몇 개가 남았을까요?",
            "T05": "{a}개짜리 {c} 묶음에서 {b}개를 빼면 몇 개일까요?",
        },
    },
    "div": {
        "train": {
            "T01": "{c} {a}개를 {b}개씩 봉지에 담으면 몇 봉지가 되고 몇 개가 남을까요?",
            "T02": "{c} {a}개를 친구 {b}명에게 똑같이 나눠 주면 한 명에 몇 개씩이고 몇 개가 남을까요?",
            "T03": "{c} {a}개를 {b}개씩 묶으면 몇 묶음이고 몇 개가 남을까요?",
        },
        "test": {
            "T04": "{c} {a}개를 상자 {b}개에 똑같이 넣으면 한 상자에 몇 개씩이고 몇 개가 남을까요?",
            "T05": "{c} {a}개를 한 접시에 {b}개씩 놓으면 몇 접시가 되고 몇 개가 남을까요?",
        },
    },
    "frac_add": {
        "train": {
            "T01": "{c}의 {x}을(를) 먹고, {y}을(를) 더 먹었어요. 모두 얼마나 먹었을까요?",
            "T02": "{c}의 {x}은(는) 내가, {y}은(는) 동생이 가졌어요. 둘이 가진 건 모두 얼마일까요?",
            "T03": "어제 {c}의 {x}, 오늘 {y}을(를) 썼어요. 모두 얼마나 썼을까요?",
        },
        "test": {
            "T04": "{c}을(를) {x}만큼 자르고 {y}만큼 더 잘랐어요. 모두 얼마나 잘랐을까요?",
            "T05": "{c}의 {x}에 {y}을(를) 더하면 얼마일까요?",
        },
    },
}

# 풀이 단계 (성취기준별). problem.steps 에 들어간다.
STEPS = {
    "[4수01-03]": ["일의 자리부터 뺀다", "위 수가 작으면 앞자리에서 10을 빌려 온다", "빌려 준 자리는 1 작아진다"],
    "[4수01-06]": ["나누는 수만큼씩 묶는다", "몇 묶음인지 센다(몫)", "남은 개수가 나누는 수보다 작은지 본다(나머지)"],
    "[4수01-15]": ["분모가 같은지 본다", "분자끼리 더한다", "분모는 그대로 둔다"],
    "[6수01-08]": ["분모가 같은지 본다", "분모를 같게 통분한다", "분자끼리 더한다"],
}


def op_of(mid: str) -> str:
    return RULES[mid][0]


def _frac_kind(standard: str) -> str:
    return "frac_same" if standard == "[4수01-15]" else "frac_diff"


def _random_operands(op: str, standard: str, split: str, rng: random.Random):
    if op == "sub":
        lo, hi = NUMBER_RANGES["sub"][split]
        a = rng.randint(lo, hi)
        b = rng.randint(10, a - 1)
        return (a, b)
    if op == "div":
        lo, hi = NUMBER_RANGES["div"][split]
        b = rng.randint(2, 9)
        a = rng.randint(lo, hi)
        return (a, b)
    if op == "frac_add":
        kind = _frac_kind(standard)
        lo, hi = NUMBER_RANGES[kind][split]
        if kind == "frac_same":
            d = rng.randint(lo, hi)
            a = rng.randint(1, d - 2)
            c = rng.randint(1, d - 1 - a)
            return ((a, d), (c, d))
        b = rng.randint(lo, hi)
        d = rng.randint(lo, hi)
        if b == d:
            return None
        a = rng.randint(1, b - 1)
        c = rng.randint(1, d - 1)
        if gcd(a, b) != 1 or gcd(c, d) != 1:  # 기약분수끼리
            return None
        if a * d + c * b >= b * d:  # 합이 1보다 작은 진분수 덧셈만
            return None
        return ((a, b), (c, d))
    raise ValueError(op)


def _valid_operands(op, operands) -> bool:
    if operands is None:
        return False
    if op == "div":
        a, b = operands
        q, r = divmod(a, b)
        return q >= 2 and r >= 1 and r != q
    if op == "sub":
        a, b = operands
        return len(str(b)) >= 2
    return True


def is_diagnostic(mid: str, operands, siblings: list[str]) -> bool:
    """오답이 정답과 다르고, 같은 성취기준의 다른 오개념 답과도 갈리는가."""
    correct = correct_answer(op_of(mid), operands)
    mine = wrong_answer(mid, operands)
    if same_value(mine, correct):
        return False
    for s in siblings:
        if s != mid and same_value(wrong_answer(s, operands), mine):
            return False
    return True


def expr(op: str, operands) -> str:
    if op == "sub":
        return f"{operands[0]} − {operands[1]}"
    if op == "div":
        return f"{operands[0]} ÷ {operands[1]}"
    (a, b), (c, d) = operands
    return f"{a}/{b} + {c}/{d}"


def word_problem(op: str, template: str, context: str, operands) -> str:
    if op == "frac_add":
        (a, b), (c, d) = operands
        return template.format(c=context, x=f"{a}/{b}", y=f"{c}/{d}")
    return template.format(c=context, a=operands[0], b=operands[1])


def make_problem(mis: dict, siblings: list[str], split: str, rng: random.Random,
                 avoid: set | None = None, context: str | None = None,
                 template_id: str | None = None) -> dict:
    mid = mis["misconception_id"]
    op = op_of(mid)
    avoid = avoid or set()
    for _ in range(10000):
        ops = _random_operands(op, mis["standard"], split, rng)
        if not _valid_operands(op, ops) or expr(op, ops) in avoid:
            continue
        if is_diagnostic(mid, ops, siblings):
            break
    else:
        raise RuntimeError(f"진단용 문제를 못 만듦: {mid}")
    tmpl_pool = TEMPLATES[op][split]
    template_id = template_id or rng.choice(sorted(tmpl_pool))
    context = context or rng.choice(CONTEXTS[op][split])
    return {
        "question": expr(op, ops),
        "answer": correct_answer(op, ops),
        "wrong_answer": wrong_answer(mid, ops),
        "steps": STEPS[mis["standard"]],
        "context": context,
        "template_id": template_id,
        "_operands": ops,
        "_word_problem": word_problem(op, tmpl_pool[template_id], context, ops),
    }
