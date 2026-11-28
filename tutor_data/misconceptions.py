"""오개념 재현 규칙 (simulator: "code").

misconceptions.jsonl 의 `rule` 을 코드로 옮긴 것과, `fix_point` 를 선생님 말에서
찾아내는 판정 규칙(코드 학생의 "알겠다" 판정)을 담는다.
"""
from __future__ import annotations

import json
from fractions import Fraction
from math import lcm
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def load_misconceptions(path: Path | None = None) -> dict[str, dict]:
    path = path or DATA_DIR / "misconceptions.jsonl"
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                d = json.loads(line)
                out[d["misconception_id"]] = d
    return out


# ---------------------------------------------------------------- 답 형식
def fmt_div(q: int, r: int) -> str:
    return f"{q}…{r}"


def fmt_frac(n: int, d: int) -> str:
    return f"{n}/{d}"


def same_value(x: str, y: str) -> bool:
    """두 답이 같은 값인지 (분수는 크기로, 나눗셈은 몫·나머지로)."""
    x, y = x.strip(), y.strip()
    if x == y:
        return True
    try:
        if "/" in x and "/" in y:
            return Fraction(x) == Fraction(y)
    except (ValueError, ZeroDivisionError):
        pass
    return False


# ---------------------------------------------------------------- 정답
def correct_answer(op: str, operands) -> str:
    if op == "sub":
        a, b = operands
        return str(a - b)
    if op == "div":
        a, b = operands
        return fmt_div(*divmod(a, b))
    if op == "frac_add":
        (a, b), (c, d) = operands
        if b == d:
            return fmt_frac(a + c, b)
        m = lcm(b, d)
        return fmt_frac(a * (m // b) + c * (m // d), m)
    raise ValueError(op)


# ---------------------------------------------------------------- 오개념 규칙
def _digits(n: int, width: int) -> list[int]:
    return [int(ch) for ch in str(n).zfill(width)][::-1]  # 일의 자리부터


def _from_digits(ds: list[int]) -> int:
    return int("".join(str(x) for x in ds[::-1]))


def sub_bigger_minus_smaller(a: int, b: int) -> str:
    """뺄셈_자리마다큰수빼기: 위아래 상관없이 각 자리에서 큰 수 − 작은 수."""
    w = len(str(a))
    da, db = _digits(a, w), _digits(b, w)
    return str(_from_digits([abs(x - y) for x, y in zip(da, db)]))


def sub_no_decrement(a: int, b: int) -> str:
    """뺄셈_빌려준자리안줄임: 10을 빌려 오지만 빌려 준 자리는 그대로 둔다."""
    w = len(str(a))
    da, db = _digits(a, w), _digits(b, w)
    res = []
    for x, y in zip(da, db):
        res.append(x + 10 - y if x < y else x - y)
    return str(_from_digits(res))


def sub_zero_borrow(a: int, b: int) -> str:
    """뺄셈_0에서받아내림: 0에서 빌릴 때 0을 9로만 바꾸고 그 앞자리는 줄이지 않는다."""
    w = len(str(a))
    da, db = _digits(a, w), _digits(b, w)
    res = []
    for i in range(w):
        if da[i] < db[i]:
            if i + 1 < w and da[i + 1] == 0:
                da[i + 1] = 9  # 0 → 9 로만 바꾸고 앞자리는 그대로
            elif i + 1 < w:
                da[i + 1] -= 1
            da[i] += 10
        res.append(da[i] - db[i])
    return str(_from_digits(res))


def div_remainder_too_big(a: int, b: int) -> str:
    """나눗셈_나머지가더큼: 몫을 작게 잡고 남은 것을 그대로 나머지로 쓴다."""
    q, r = divmod(a, b)
    return fmt_div(q - 1, r + b)


def div_swap(a: int, b: int) -> str:
    """나눗셈_몫나머지바꿈: 몫 자리에 나머지, 나머지 자리에 몫을 쓴다."""
    q, r = divmod(a, b)
    return fmt_div(r, q)


def frac_add_denominators(x, y) -> str:
    """분수덧셈_분모도더함 / 분수덧셈_통분없이위아래더함: 위끼리, 아래끼리 더한다."""
    (a, b), (c, d) = x, y
    return fmt_frac(a + c, b + d)


def frac_numerators_only(x, y) -> str:
    """분수덧셈_분자만더함: 분자만 더하고 분모는 둘 중 큰 것을 쓴다."""
    (a, b), (c, d) = x, y
    return fmt_frac(a + c, max(b, d))


# misconception_id → (연산, 규칙 함수)
RULES = {
    "뺄셈_자리마다큰수빼기": ("sub", sub_bigger_minus_smaller),
    "뺄셈_빌려준자리안줄임": ("sub", sub_no_decrement),
    "뺄셈_0에서받아내림": ("sub", sub_zero_borrow),
    "나눗셈_나머지가더큼": ("div", div_remainder_too_big),
    "나눗셈_몫나머지바꿈": ("div", div_swap),
    "분수덧셈_분모도더함": ("frac_add", frac_add_denominators),
    "분수덧셈_통분없이위아래더함": ("frac_add", frac_add_denominators),
    "분수덧셈_분자만더함": ("frac_add", frac_numerators_only),
}


def wrong_answer(mid: str, operands) -> str:
    op, fn = RULES[mid]
    return fn(*operands)


# ---------------------------------------------------------------- fix_point 판정
# 각 묶음(그룹)에서 하나 이상이 선생님 말에 있으면 fix_point 를 짚은 것으로 본다.
FIX_POINT_KEYWORDS: dict[str, list[list[str]]] = {
    # 위 수가 작으면 앞자리에서 10을 빌려 온다
    "뺄셈_자리마다큰수빼기": [["빌려", "빌리", "받아내림", "가져오", "가져와"]],
    # 빌려 준 자리는 1 작아진다
    "뺄셈_빌려준자리안줄임": [
        ["빌려 준", "빌려준", "빌려 간", "빌려간", "빌려 주", "빌려주", "빌려 줬", "빌려줬", "내준", "내 준", "내줬"],
        ["1 작아", "1이 작아", "하나 작아", "1 줄", "1이 줄", "하나 줄", "줄어", "작아져", "작아지"],
    ],
    # 0은 빌려 줄 게 없어서 앞자리에서 먼저 빌려 와야 한다
    "뺄셈_0에서받아내림": [
        ["빌려 줄 게 없", "빌려줄 게 없", "빌려 줄 수 없", "빌려줄 수 없", "줄 게 없", "줄 수 없",
         "앞자리에서 먼저", "먼저 빌려", "백의 자리에서 빌려", "백의 자리에서 먼저"],
    ],
    # 나머지가 나누는 수보다 크면 한 묶음 더 만들 수 있다
    "나눗셈_나머지가더큼": [
        ["나머지", "남은", "남는"],
        ["한 묶음 더", "더 묶", "더 만들", "더 나눌", "더 담", "나누는 수보다 크", "보다 커", "보다 크면"],
    ],
    # 몫 = 몇 묶음, 나머지 = 묶고 남은 개수
    "나눗셈_몫나머지바꿈": [
        ["몫"],
        ["묶음", "몇 번", "몇 묶", "몇 개씩"],
        ["나머지"],
        ["남은", "남는", "남아"],
    ],
    # 조각 크기(분모)는 그대로다
    "분수덧셈_분모도더함": [
        ["조각 크기", "조각의 크기", "분모", "아래 숫자", "아랫 숫자", "아래 수"],
        ["그대로", "안 바뀌", "바뀌지 않", "변하지 않", "안 변", "똑같", "같아"],
    ],
    # 조각 크기가 다르면 먼저 같은 크기로 맞춘다
    "분수덧셈_통분없이위아래더함": [
        ["조각 크기", "조각의 크기", "분모", "크기"],
        ["같은 크기로", "맞춰", "맞추", "맞춘", "통분", "똑같이 만들", "같게"],
    ],
    "분수덧셈_분자만더함": [
        ["조각 크기", "조각의 크기", "분모", "크기"],
        ["같은 크기로", "맞춰", "맞추", "맞춘", "통분", "똑같이 만들", "같게"],
    ],
}


def hits_fix_point(mid: str, teacher_text: str) -> bool:
    t = teacher_text.replace(" ", " ")
    return all(any(k in t for k in group) for group in FIX_POINT_KEYWORDS[mid])


# ---------------------------------------------------------------- 아이 말 (오개념 믿음 / 이해)
# 계산 문제가 아닌 질문(개념 질문)에 대한 답. 교정 전에는 오개념 규칙대로, 교정 후에는 fix_point 대로.
BELIEF = {
    "뺄셈_자리마다큰수빼기": ["큰 수에서 작은 수 빼면 되잖아요", "위가 작으면 아래에서 위를 빼면 돼요"],
    "뺄셈_빌려준자리안줄임": ["10 빌려 왔으니까 그냥 빼면 돼요", "빌려 준 자리는 그대로 아니에요?"],
    "뺄셈_0에서받아내림": ["0을 9로 바꾸면 되잖아요", "0이 9가 되니까 앞은 그대로예요"],
    "나눗셈_나머지가더큼": ["남은 거 다 나머지 아니에요?", "남은 게 나머지니까 그대로 써요"],
    "나눗셈_몫나머지바꿈": ["남은 게 앞에 오는 거 아니에요?", "앞에가 남은 거고 뒤에가 묶음이에요"],
    "분수덧셈_분모도더함": ["위도 더하고 아래도 더하면 돼요", "아래 숫자도 더해야 하잖아요"],
    "분수덧셈_통분없이위아래더함": ["위끼리 더하고 아래끼리 더하면 돼요", "아래 숫자도 더하면 되잖아요"],
    "분수덧셈_분자만더함": ["위만 더하고 아래는 큰 거 쓰면 돼요", "아래는 큰 숫자 쓰면 되잖아요"],
}

UNDERSTAND = {
    # [0] 알아듣는 순간 ("아, …구나"), [1] 이유를 물으면 하는 설명
    "뺄셈_자리마다큰수빼기": ["아, 위가 작으면 앞에서 10을 빌려 와야 하는구나", "위가 작으면 앞자리에서 10을 빌려 와요"],
    "뺄셈_빌려준자리안줄임": ["아, 빌려 준 자리는 1 작아지는구나", "빌려 줬으니까 그 자리는 하나 줄어요"],
    "뺄셈_0에서받아내림": ["아, 0은 빌려 줄 게 없어서 앞에서 먼저 빌려 와야 하는구나", "0은 빌려 줄 게 없어서 앞자리에서 먼저 빌려 와요"],
    "나눗셈_나머지가더큼": ["아, 남은 게 나누는 수보다 크면 한 묶음 더 만들 수 있구나", "나머지는 나누는 수보다 작아야 해요"],
    "나눗셈_몫나머지바꿈": ["아, 몫은 몇 묶음이고 나머지는 남은 거구나", "묶음 수가 몫이고 남은 게 나머지예요"],
    "분수덧셈_분모도더함": ["아, 조각 크기는 그대로구나", "조각 크기가 안 바뀌니까 아래 숫자는 그대로예요"],
    "분수덧셈_통분없이위아래더함": ["아, 조각 크기를 먼저 똑같이 맞춰야 하는구나", "조각 크기가 다르면 먼저 같은 크기로 맞춰요"],
    "분수덧셈_분자만더함": ["아, 조각 크기를 먼저 똑같이 맞춰야 하는구나", "조각 크기가 다르면 먼저 같은 크기로 맞춰요"],
}
