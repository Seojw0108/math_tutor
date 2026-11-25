"""평가 하네스의 학생 역할 (⑤).

  코드 학생   A가 만드는 모듈 (수와 연산). 도착 전까지는 RuleStudent 가 같은 인터페이스로 대신한다.
  AI 학생     LLMStudent (전 영역)
  사람        HumanStudent (터미널에서 직접 아이 역할, 20회)

학생 인터페이스 (A의 코드 학생도 이 메서드만 맞추면 하네스에 그대로 꽂힌다):

    class Student:
        understood: bool                                   # 오개념이 풀렸는가
        def reset(self, setting: dict, seed: int) -> None   # setting = dialogues.jsonl 한 줄(test)의 설정 부분
        def respond(self, messages: list[dict]) -> dict     # 선생님 말까지의 대화 -> 아이 턴
            # {"content": str, "response_type": str, "expected_answer": str|None, "understood": bool}
        def answer_post_check(self, question: str) -> str   # 사후 문제 답

A 모듈 연결: --student code --student_module path.to.module:ClassName  (load_student 참고)
"""
from __future__ import annotations

import importlib
import random
import re
from fractions import Fraction

from .chat import transcript
from .llm_client import LLMClient
from .prompts import STUDENT_SYSTEM, student_post_check_prompt, student_turn_prompt


# ---------------------------------------------------------------------------
# 확정 오개념 8개의 '아이 머릿속 규칙'
# ---------------------------------------------------------------------------
def _digits(n: int, width: int) -> list[int]:
    return [int(c) for c in str(n).zfill(width)][::-1]  # 일의 자리부터


def sub_bigminussmall(a: int, b: int) -> int:
    w = max(len(str(a)), len(str(b)))
    da, db = _digits(a, w), _digits(b, w)
    return int("".join(str(abs(x - y)) for x, y in zip(da, db))[::-1])


def sub_borrow_no_decrement(a: int, b: int) -> int:
    w = max(len(str(a)), len(str(b)))
    da, db = _digits(a, w), _digits(b, w)
    out = []
    for x, y in zip(da, db):
        out.append(x + 10 - y if x < y else x - y)  # 빌려 오지만 빌려 준 자리는 그대로
    return int("".join(str(d) for d in out[::-1]))


def sub_zero_borrow(a: int, b: int) -> int:
    w = max(len(str(a)), len(str(b)))
    da, db = _digits(a, w), _digits(b, w)
    out = []
    for i in range(w):
        x, y = da[i], db[i]
        if x < y:
            x += 10
            j = i + 1
            if j < w:
                if da[j] == 0:
                    da[j] = 9          # 0을 9로만 바꾸고 그 앞자리는 줄이지 않는다
                else:
                    da[j] -= 1
        out.append(x - y)
    return int("".join(str(d) for d in out[::-1]))


def div_big_remainder(a: int, b: int) -> tuple[int, int]:
    q = max(0, a // b - 1)
    return q, a - q * b


def div_swap(a: int, b: int) -> tuple[int, int]:
    q, r = divmod(a, b)
    return r, q


def frac_add_both(x: Fraction, y: Fraction, raw: tuple) -> str:
    (a, b), (c, d) = raw
    return f"{a + c}/{b + d}"


def frac_add_num_only(x: Fraction, y: Fraction, raw: tuple) -> str:
    (a, b), (c, d) = raw
    return f"{a + c}/{max(b, d)}"


RULES = {
    "뺄셈_자리마다큰수빼기": ("sub", sub_bigminussmall),
    "뺄셈_빌려준자리안줄임": ("sub", sub_borrow_no_decrement),
    "뺄셈_0에서받아내림": ("sub", sub_zero_borrow),
    "나눗셈_나머지가더큼": ("div", div_big_remainder),
    "나눗셈_몫나머지바꿈": ("div", div_swap),
    "분수덧셈_분모도더함": ("fadd", frac_add_both),
    "분수덧셈_통분없이위아래더함": ("fadd", frac_add_both),
    "분수덧셈_분자만더함": ("fadd", frac_add_num_only),
}

# 선생님이 이 낱말로 핵심을 짚으면 '알겠다' 쪽으로 간다 (fix_point 기반)
FIX_KEYWORDS = {
    "뺄셈_자리마다큰수빼기": ["빌려", "받아내림", "위 수가 작", "위의 수가 작", "뺄 수 없", "뺄 수 있"],
    "뺄셈_빌려준자리안줄임": ["빌려 준", "빌려준", "1 작아", "하나 줄", "1 줄", "줄어들"],
    "뺄셈_0에서받아내림": ["빌려 줄 게 없", "0은 빌려", "앞자리에서", "백의 자리에서", "먼저 빌려"],
    "나눗셈_나머지가더큼": ["나머지가 나누는 수보다", "한 묶음 더", "한 명에게 더", "더 나눠 줄", "더 줄 수"],
    "나눗셈_몫나머지바꿈": ["몇 묶음", "몫은", "묶고 남은", "남은 개수", "남은 것"],
    "분수덧셈_분모도더함": ["조각 크기", "분모는 그대로", "아래 숫자는 그대로", "크기가 바뀌", "크기는 그대로"],
    "분수덧셈_통분없이위아래더함": ["같은 크기", "크기가 다르", "통분", "분모를 같게", "크기를 맞"],
    "분수덧셈_분자만더함": ["같은 크기", "크기가 다르", "통분", "분모를 같게", "크기를 맞"],
}

_INT = r"\d{1,4}"
_SUB = re.compile(rf"({_INT})\s*[−\-–]\s*({_INT})(?!\s*/)")
_DIV = re.compile(rf"({_INT})\s*÷\s*({_INT})")
_FADD = re.compile(r"(\d+)\s*/\s*(\d+)\s*\+\s*(\d+)\s*/\s*(\d+)")
_FRAC = re.compile(r"(\d+)\s*/\s*(\d+)")
_NUMS = re.compile(r"\d+")


def parse_problem(text: str, kind: str) -> tuple | None:
    """선생님 말에서 이 오개념 종류의 문제를 찾는다. 식이 없으면 문장제 숫자로 추정."""
    if kind == "sub":
        m = _SUB.search(text)
        if m:
            return int(m.group(1)), int(m.group(2))
        if any(w in text for w in ("남", "빼", "덜", "더 해야", "차이")):
            nums = [int(n) for n in _NUMS.findall(text) if not re.search(rf"{n}\s*/", text)]
            if len(nums) >= 2 and nums[0] > nums[1] >= 10:
                return nums[0], nums[1]
    elif kind == "div":
        m = _DIV.search(text)
        if m:
            return int(m.group(1)), int(m.group(2))
        if any(w in text for w in ("나눠", "나누", "묶", "씩")):
            nums = [int(n) for n in _NUMS.findall(text)]
            if len(nums) >= 2 and nums[0] > nums[1] > 1:
                return nums[0], nums[1]
    elif kind == "fadd":
        m = _FADD.search(text)
        if m:
            a, b, c, d = map(int, m.groups())
            return (a, b), (c, d)
        fr = _FRAC.findall(text)
        if len(fr) >= 2 and any(w in text for w in ("더", "합", "모두", "얼마")):
            (a, b), (c, d) = [tuple(map(int, f)) for f in fr[:2]]
            return (a, b), (c, d)
    return None


def solve(kind: str, prob: tuple, rule=None) -> str:
    if kind == "sub":
        a, b = prob
        return str(rule(a, b) if rule else a - b)
    if kind == "div":
        a, b = prob
        q, r = rule(a, b) if rule else divmod(a, b)
        return f"{q}…{r}"
    (a, b), (c, d) = prob
    x, y = Fraction(a, b), Fraction(c, d)
    if rule:
        return rule(x, y, prob)
    s = x + y
    common = b if b == d else None
    if common and (s * common).denominator == 1:  # 같은 분모면 약분하지 않은 꼴 (1/4+1/4=2/4)
        return f"{int(s * common)}/{common}"
    return f"{s.numerator}/{s.denominator}"


_ANSWER_STYLE = {
    "짧게 대답": ["{a}", "{a}요"],
    "수다스러움": ["음 제가 해 봤는데요, {a}요!", "{a}요! 맞죠? 저 이거 잘해요"],
    "소심함": ["{a}...요?", "음... {a}... 맞아요?"],
    "장난스러움": ["{a}! 쉽다", "{a}요 히히"],
}
_SLANG = ["ㅋㅋ", "ㄹㅇ", "개쉽네", "킹받네"]


class RuleStudent:
    """A의 코드 학생이 오기 전까지 쓰는 대체 학생 (확정 오개념 8개). 동작은 단순하다:

    * 선생님 말에서 해당 종류의 문제(식/문장제)를 찾으면 '머릿속 규칙'으로 푼다 (알겠다 전에는 오답).
    * 선생님이 핵심(FIX_KEYWORDS)을 질문으로 짚으면 +1, 설명으로만 말하면 +0.5. 1.5 이상이면 understood.
    * 선생님이 정답만 알려 주면 따라 말할 수는 있지만 understood 는 바뀌지 않는다.
    """

    def __init__(self, slang_rate: float = 0.1):
        self.slang_rate = slang_rate

    def reset(self, setting: dict, seed: int = 0):
        self.setting = setting
        self.mid = setting["misconception_id"]
        if self.mid not in RULES:
            raise KeyError(f"RuleStudent does not know {self.mid}; use the code student from A or LLMStudent")
        self.kind, self.rule = RULES[self.mid]
        self.persona = setting.get("student_persona") or {"style": "짧게 대답", "slang": False}
        self.rng = random.Random(seed)
        self.progress = 0.0
        self.understood = False
        self.shown_misconception = False

    def _style(self, ans: str) -> str:
        s = self.rng.choice(_ANSWER_STYLE.get(self.persona.get("style"), _ANSWER_STYLE["짧게 대답"])).format(a=ans)
        if self.persona.get("slang") and self.rng.random() < self.slang_rate:
            s += " " + self.rng.choice(_SLANG)
        return s

    def _update_understanding(self, teacher: str):
        t = teacher.replace(" ", "")
        if any(k.replace(" ", "") in t for k in FIX_KEYWORDS.get(self.mid, [])):
            self.progress += 1.0 if teacher.rstrip().endswith("?") else 0.5
        if self.shown_misconception and self.progress >= 1.5:
            self.understood = True

    def respond(self, messages: list[dict]) -> dict:
        teacher = messages[-1]["content"]
        self._update_understanding(teacher)
        prob = parse_problem(teacher, self.kind)
        if prob is not None:
            correct = solve(self.kind, prob)
            ans = correct if self.understood else solve(self.kind, prob, self.rule)
            rt = "정답" if ans == correct else "오개념오답"
            if rt == "오개념오답":
                self.shown_misconception = True
            content = self._style(ans)
            if self.persona.get("slang") and any(s in content for s in _SLANG):
                rt = "유행어" if rt == "정답" else rt
            return {"content": content, "response_type": rt, "expected_answer": correct,
                    "understood": self.understood}
        # 계산 문제가 아닌 질문. 아직 모를 때 가끔 기획서의 아이 말투("몰라요", "어쩔", "그냥 답 알려주세요")
        t = teacher.rstrip()
        if not self.understood and t.endswith("?") and self.rng.random() < 0.15:
            reply, rt = self.rng.choice([("몰라요", "모름"), ("그냥 답 알려주세요", "답요구")]
                                        + ([("어쩔", "유행어")] if self.persona.get("slang") else []))
            return {"content": reply, "response_type": rt, "expected_answer": None, "understood": False}
        m = re.search(r"([가-힣0-9]+)(?:이에요|예요|일까요)?,\s*([가-힣0-9]+)(?:이에요|예요|일까요)\?$", t)
        if m and self.shown_misconception:
            pick = m.group(2) if self.progress >= 1 else m.group(1)
            return {"content": self._style(pick), "response_type": "정답" if self.progress >= 1 else "모름",
                    "expected_answer": None, "understood": self.understood}
        if t.endswith("?"):
            reply = self.rng.choice(["네", "음... 잘 모르겠어요", "어... 그런가요?", "네!"]) if not self.understood \
                else self.rng.choice(["아! 알 것 같아요", "네, 이제 알겠어요"])
            rt = "모름" if "모르" in reply else "정답"
            return {"content": reply, "response_type": rt, "expected_answer": None, "understood": self.understood}
        return {"content": self.rng.choice(["네", "네!", "알겠어요"]), "response_type": "정답",
                "expected_answer": None, "understood": self.understood}

    def is_diagnostic(self, question: str) -> bool | None:
        """이 문제에서 오개념 규칙과 바른 풀이의 답이 갈리는가 (안 갈리면 교정 여부를 판정할 수 없다)."""
        prob = parse_problem(question, self.kind)
        if prob is None:
            return None
        return solve(self.kind, prob) != solve(self.kind, prob, self.rule)

    def answer_post_check(self, question: str) -> str:
        prob = parse_problem(question, self.kind)
        if prob is None:
            return "모르겠어요"
        return solve(self.kind, prob) if self.understood else solve(self.kind, prob, self.rule)


class LLMStudent:
    """AI 학생 (전 영역). 오개념 정보는 misconceptions.jsonl 에서."""

    def __init__(self, misconceptions: dict, client: LLMClient | None = None):
        self.mcs = misconceptions
        self.client = client or LLMClient(prefix="STUDENT", temperature=0.7, max_tokens=200)

    def reset(self, setting: dict, seed: int = 0):
        self.setting = setting
        self.mc = self.mcs.get(setting["misconception_id"], {"name": setting["misconception_id"]})
        self.persona = setting.get("student_persona") or {}
        self.seed = seed
        self.understood = False
        self.turn = 0

    def respond(self, messages: list[dict]) -> dict:
        self.turn += 1
        p = student_turn_prompt(self.setting.get("grade_band", ""), self.persona, self.mc,
                                transcript(messages), self.understood, bool(self.persona.get("slang")))
        obj = self.client.chat_json(STUDENT_SYSTEM, p, seed=self.seed * 100 + self.turn)
        self.understood = self.understood or bool(obj.get("understood"))
        return {"content": str(obj.get("content", "")).strip() or "음...", "response_type": obj.get("response_type"),
                "expected_answer": None, "understood": self.understood}

    def answer_post_check(self, question: str) -> str:
        p = student_post_check_prompt(self.setting.get("grade_band", ""), self.mc, self.understood, question)
        return str(self.client.chat_json(STUDENT_SYSTEM, p, seed=self.seed).get("answer", ""))


class HumanStudent:
    """사람이 아이 역할 (터미널). response_type 은 사람이 골라 적거나 비워 둔다."""

    def reset(self, setting: dict, seed: int = 0):
        self.setting = setting
        self.understood = False
        print(f"\n=== 설정: {setting.get('grade_band')} / {setting.get('standard_name')} / 오개념 {setting.get('misconception_id')} ===")

    def respond(self, messages: list[dict]) -> dict:
        print(f"\n선생님: {messages[-1]['content']}")
        content = input("아이> ").strip()
        if content == "/알겠다":
            self.understood = True
            content = input("아이(알겠다 표시됨)> ").strip()
        return {"content": content, "response_type": None, "expected_answer": None, "understood": self.understood}

    def answer_post_check(self, question: str) -> str:
        return input(f"\n[사후 문제] {question}\n아이 답> ").strip()


def load_student(kind: str, misconceptions: dict | None = None, module: str | None = None):
    """kind: code | rule | llm | human. code 는 A 모듈(module='pkg.mod:Class'), 없으면 RuleStudent."""
    if kind == "code":
        if module:
            mod, cls = module.split(":")
            return getattr(importlib.import_module(mod), cls)()
        return RuleStudent()
    if kind == "rule":
        return RuleStudent()
    if kind == "llm":
        return LLMStudent(misconceptions or {})
    if kind == "human":
        return HumanStudent()
    raise ValueError(kind)
