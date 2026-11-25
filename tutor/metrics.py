"""지표 함수 (③). 필터(A)와 최종 평가 지표(B)가 같은 함수다.

입력: 대화 1건(dict, dialogues.jsonl 한 줄) -> 출력: filters 칸에 들어갈 dict.

    from tutor.metrics import compute_filters
    d["filters"] = compute_filters(d, judge_fn=None)          # judge 없이 (빠른 1차 필터)
    d["filters"] = compute_filters(d, judge_fn=my_judge)      # judge 포함 (최종)

필드
  corrected        사후 문제(post_check)를 아이가 맞혔는가 (데이터의 corrected 가 있으면 그대로)
  calc_ok          선생님 말 속 계산식(분수·사칙연산·나머지 나눗셈)을 정규식으로 뽑아 검산. 하나라도 틀리면 False
  easy_words       선생님 말의 내용어(명사·동사·형용사·부사) 중 초등 기본 어휘 + 학년군에 맞는 수학 용어 비율
  avg_sentence_len 선생님 말의 평균 문장 길이 (글자 수, 띄어쓰기 포함)
  question_ratio   물음표로 끝나는 선생님 턴 비율
  early_answer     첫 피드백 턴(아이 첫 오답 바로 뒤 선생님 말)에 problem.answer 가 나오는가
  safety_ok        유행어·비속어 따라 하기 / 비속어 사용 / 개인정보 질문이 없는가 (목록 대조 [+ AI 판정])
  judge            AI 심사 루브릭 6항목 평균 (judge_fn 이 있을 때만, 없으면 None)
  duplicate        데이터셋 단위 중복 (mark_duplicates 가 채움, 대화 1건 단독으로는 False)
  passed           위 기준을 모두 통과 (기준값: THRESHOLDS)

의존성: 표준 라이브러리만. kiwipiepy 가 깔려 있으면 형태소 분석으로 easy_words 를 더 정확히 잰다
(없으면 조사·어미를 떼는 규칙 기반으로 대신한다. 두 방식의 값은 조금 다르므로 팀 전체가 같은 방식을 쓴다).
"""
from __future__ import annotations

import hashlib
import re
from fractions import Fraction
from functools import lru_cache
from pathlib import Path
from typing import Callable, Iterable

RES = Path(__file__).resolve().parent / "resources"

# ---------------------------------------------------------------------------
# 기준값 (② Base 측정 후 제안 -> A와 함께 확정). scripts/base_report.py 가 제안값을 계산한다.
# ---------------------------------------------------------------------------
THRESHOLDS = {
    "easy_words_min": 0.90,
    "avg_sentence_len_max": 25.0,
    "question_ratio_min": 0.50,
    "judge_min": 3.5,
}
# 학년군별 말투 기준 (기획서: 1학년은 더 쉽게, 6학년은 조금 더 정확하게). 없는 키는 THRESHOLDS 값.
THRESHOLDS_BY_BAND = {
    "1~2": {"easy_words_min": 0.92, "avg_sentence_len_max": 20.0},
    "3~4": {"easy_words_min": 0.90, "avg_sentence_len_max": 25.0},
    "5~6": {"easy_words_min": 0.88, "avg_sentence_len_max": 30.0},
}


def thresholds_for(grade_band: str | None, overrides: dict | None = None) -> dict:
    return {**THRESHOLDS, **THRESHOLDS_BY_BAND.get(grade_band or "", {}), **(overrides or {})}


WRONG_TYPES = {"오개념오답", "계산실수"}
GRADE_ORDER = {"1~2": 1, "3~4": 2, "5~6": 3}


# ---------------------------------------------------------------------------
# 공용
# ---------------------------------------------------------------------------
def teacher_turns(dialogue: dict) -> list[str]:
    return [m["content"] for m in dialogue.get("messages", []) if m["role"] == "assistant"]


def _strip_think(text: str) -> str:
    if "</think>" in text:
        text = text.split("</think>", 1)[1]
    return text.strip()


SENT_SPLIT = re.compile(r"(?<=[.?!])\s+|\n+|(?<=[.?!][\"'”’)])\s*")


def sentences(text: str) -> list[str]:
    text = _strip_think(text)
    # 소수점(0.5)은 뒤에 공백이 없어서 나뉘지 않는다
    return [p.strip() for p in SENT_SPLIT.split(text) if p and p.strip()]


def norm_answer(s: str) -> str:
    s = str(s).strip().replace(" ", "")
    s = s.replace("...", "…").replace("..", "…")
    return s.rstrip(".요예이에").strip()


# ---------------------------------------------------------------------------
# 어휘 (easy_words)
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def load_basic_words(path: str | None = None) -> frozenset[str]:
    p = Path(path) if path else RES / "basic_words.txt"
    words: set[str] = set()
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0]
        words.update(w for w in line.split() if w)
    return frozenset(words)


@lru_cache(maxsize=1)
def load_math_terms(path: str | None = None) -> dict[str, int]:
    p = Path(path) if path else RES / "math_terms.tsv"
    terms: dict[str, int] = {}
    for line in p.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        term, band = line.split("\t")[:2]
        terms[term.strip()] = GRADE_ORDER[band.strip()]
    return terms


@lru_cache(maxsize=1)
def _kiwi():
    try:
        from kiwipiepy import Kiwi  # type: ignore
        return Kiwi()
    except Exception:  # pragma: no cover - 선택 의존성
        return None


@lru_cache(maxsize=1)
def _warn_no_kiwi():
    import warnings
    warnings.warn("kiwipiepy 가 없어 규칙 기반으로 easy_words 를 잽니다. 값이 달라지니 `pip install kiwipiepy` 하세요.")


CONTENT_TAGS = {"NNG", "NNP", "VV", "VA", "MAG", "XR"}

# kiwi 가 없을 때 쓰는 조사·어미 목록 (긴 것부터 뗀다)
_SUFFIXES = sorted(set("""
이에요 예요 이었어요 였어요 이라고 라고 이랑 랑 에서는 에서 에게 한테 께서 으로 로 부터 까지 마다 처럼 보다 하고
은 는 이 가 을 를 의 에 와 과 도 만 요 야 아 어 지 네 죠 까 고 며 면 서 니
어요 아요 해요 했어요 했지요 할까요 일까요 을까요 ㄹ까요 볼까요 해볼까요 봐요 줘요 세요 습니다 합니다 입니다 했구나 구나 군요 겠지
""".split()), key=len, reverse=True)


def content_words(text: str) -> list[tuple[str, str]]:
    """(낱말, 품사) 목록. 동사·형용사는 '어간+다' 기본형으로."""
    text = _strip_think(text)
    kiwi = _kiwi()
    out: list[tuple[str, str]] = []
    if kiwi is not None:
        for tok in kiwi.tokenize(text):
            tag = tok.tag.split("-")[0]
            if tag not in CONTENT_TAGS:
                continue
            form = tok.form + "다" if tag in ("VV", "VA") else tok.form
            out.append((form, tag))
        return out
    # 규칙 기반 대체 (정확도가 낮다: 팀 전체가 kiwipiepy 를 깔고 쓸 것)
    _warn_no_kiwi()
    for eo in re.findall(r"[가-힣]+", text):
        w = eo
        for _ in range(2):
            for suf in _SUFFIXES:
                if len(w) > len(suf) and w.endswith(suf):
                    w = w[: -len(suf)]
                    break
        out.append((w, "?"))
    return out


def word_is_easy(word: str, grade_band: str | None, basic: frozenset[str], terms: dict[str, int]) -> bool:
    if word in basic:
        return True
    g = GRADE_ORDER.get(grade_band or "", 3)
    t = terms.get(word)
    if t is not None:
        return t <= g
    if word.endswith("다") and word[:-1] + "하다" in basic:
        return True
    # 규칙 기반 대체일 때: 목록 낱말로 시작하면 쉬운 말로 친다 (2글자 이상만)
    return any(len(b) >= 2 and word.startswith(b) for b in basic) if len(word) >= 3 else False


def easy_words(texts: Iterable[str], grade_band: str | None = None, return_oov: bool = False):
    basic, terms = load_basic_words(), load_math_terms()
    total = easy = 0
    oov: list[str] = []
    for t in texts:
        for w, _tag in content_words(t):
            if re.fullmatch(r"[0-9a-zA-Z]+", w):
                continue
            total += 1
            if word_is_easy(w, grade_band, basic, terms):
                easy += 1
            else:
                oov.append(w)
    ratio = round(easy / total, 4) if total else 1.0
    return (ratio, oov) if return_oov else ratio


def avg_sentence_len(texts: Iterable[str]) -> float:
    sents = [s for t in texts for s in sentences(t)]
    return round(sum(len(s) for s in sents) / len(sents), 2) if sents else 0.0


# ---------------------------------------------------------------------------
# 질문 비율
# ---------------------------------------------------------------------------
_TRAIL_CHARS = " \t\n\"'”’)]~^.…♡♥☆★✨⭐👍👏😊🙂😄😀🤔💡🍕️"


def ends_with_question(text: str) -> bool:
    t = _strip_think(text).rstrip(_TRAIL_CHARS)
    return t.endswith("?") or t.endswith("?!")


def question_ratio(texts: list[str]) -> float:
    return round(sum(ends_with_question(t) for t in texts) / len(texts), 4) if texts else 0.0


# ---------------------------------------------------------------------------
# 계산 검산 (calc_ok)
# ---------------------------------------------------------------------------
_NUM = r"\d+(?:\.\d+)?(?:\s*/\s*\d+)?"
_UNIT = r"(?:cm|mm|km|m|kg|g|mL|ml|L|t|원|개|명|분|초|시간|cm²|m²|cm³|°)?"
_OP = r"[+\-−–×xX*÷]"
_EXPR = re.compile(
    rf"(?<![\d/.])({_NUM}){_UNIT}((?:\s*{_OP}\s*{_NUM}{_UNIT})+)\s*=\s*({_NUM}){_UNIT}"
    rf"(?:\s*(?:…|\.\.\.|나머지\s*)\s*(\d+))?"
)
_TERM = re.compile(rf"\s*({_OP})\s*({_NUM})")


def _to_frac(s: str) -> Fraction:
    s = s.replace(" ", "")
    if "/" in s:
        a, b = s.split("/")
        if int(b) == 0:
            raise ZeroDivisionError
        return Fraction(Fraction(a), Fraction(b))
    return Fraction(s)


def _eval(first: str, rest: str) -> Fraction | None:
    vals = [_to_frac(first)]
    ops: list[str] = []
    for op, num in _TERM.findall(rest):
        ops.append({"−": "-", "–": "-", "x": "*", "X": "*", "×": "*", "÷": "/"}.get(op, op))
        vals.append(_to_frac(num))
    # × ÷ 먼저
    i = 0
    while i < len(ops):
        if ops[i] in "*/":
            if ops[i] == "/" and vals[i + 1] == 0:
                return None
            v = vals[i] * vals[i + 1] if ops[i] == "*" else vals[i] / vals[i + 1]
            vals[i:i + 2] = [v]
            ops.pop(i)
        else:
            i += 1
    total = vals[0]
    for op, v in zip(ops, vals[1:]):
        total = total + v if op == "+" else total - v
    return total


def check_calculations(text: str, ignore_values: set[str] | None = None) -> list[dict]:
    """텍스트 속 'a op b = c' 꼴 계산식을 모두 검산. 반환: [{expr, ok}]

    ignore_values: 아이가 말한 오답(예: '2/8'). 선생님이 아이 오답을 되짚는 식
    ('2/8 이라고 했구나', '50-23=33 이라고 생각했구나')은 계산 오류로 치지 않는다.
    """
    ignore = {norm_answer(v) for v in (ignore_values or set())}
    results = []
    for m in _EXPR.finditer(_strip_think(text)):
        first, rest, rhs, rem = m.group(1), m.group(2), m.group(3), m.group(4)
        expr = m.group(0)
        rhs_key = norm_answer(rhs + (f"…{rem}" if rem else ""))
        if rhs_key in ignore or norm_answer(rhs) in ignore:
            continue
        try:
            if rem is not None:  # 나머지 있는 나눗셈: a ÷ b = q…r
                terms = _TERM.findall(rest)
                if len(terms) != 1 or terms[0][0] != "÷":
                    continue
                a, b = int(_to_frac(first)), int(_to_frac(terms[0][1]))
                ok = b != 0 and divmod(a, b) == (int(_to_frac(rhs)), int(rem))
            else:
                val = _eval(first, rest)
                if val is None:
                    continue
                ok = val == _to_frac(rhs)
                # 소수 반올림 표기 허용 (예: 10÷3=3.33)
                if not ok and "." in rhs:
                    ok = abs(float(val) - float(rhs)) < 10 ** -(len(rhs.split(".")[1])) + 1e-9
        except (ValueError, ZeroDivisionError):
            continue
        results.append({"expr": expr.strip(), "ok": bool(ok)})
    return results


def child_said_values(dialogue: dict) -> set[str]:
    vals: set[str] = set()
    for m in dialogue.get("messages", []):
        if m["role"] == "user":
            vals.update(re.findall(rf"{_NUM}(?:\s*(?:…|\.\.\.)\s*\d+)?", m["content"]))
    wa = (dialogue.get("problem") or {}).get("wrong_answer")
    if wa:
        vals.add(str(wa))
    return vals


def calc_ok(dialogue: dict, return_details: bool = False):
    ignore = child_said_values(dialogue)
    details = [r for t in teacher_turns(dialogue) for r in check_calculations(t, ignore)]
    ok = all(r["ok"] for r in details)
    return (ok, details) if return_details else ok


# ---------------------------------------------------------------------------
# 바로 정답 말하기 (early_answer)
# ---------------------------------------------------------------------------
def answer_variants(answer: str) -> list[str]:
    a = str(answer).strip()
    out = {a, a.replace(" ", "")}
    m = re.fullmatch(r"(\d+)\s*(?:…|\.\.\.|나머지)\s*(\d+)", a)
    if m:
        q, r = m.groups()
        out |= {f"{q}…{r}", f"{q}...{r}", f"{q} 나머지 {r}", f"{q}나머지{r}",
                f"몫은 {q}", f"몫이 {q}", f"몫: {q}"}
    return [v for v in out if v]


def contains_answer(text: str, answer: str | None) -> bool:
    if not answer:
        return False
    t = _strip_think(text)
    for v in answer_variants(answer):
        if re.search(r"[0-9]", v):
            # 숫자 답: 앞뒤가 숫자·분수선·소수점이 아니어야 (2/4 가 12/45 안에서 잡히지 않게)
            core = re.escape(v).replace(r"\ ", r"\s*").replace("/", r"\s*/\s*")
            pat = rf"(?<![\d/.]){core}(?![\d/]|\.\d)"
            if re.search(pat, t):
                return True
        elif len(v) >= 2 and v in t:
            return True
    return False


def first_feedback_index(messages: list[dict], wrong_answer: str | None = None) -> int | None:
    """아이 첫 오답 바로 뒤 선생님 턴의 index (messages 기준)."""
    for i, m in enumerate(messages):
        if m["role"] != "user" or i + 1 >= len(messages) or messages[i + 1]["role"] != "assistant":
            continue
        rt = m.get("response_type")
        if rt in WRONG_TYPES:
            return i + 1
        if rt is None and wrong_answer and contains_answer(m["content"], wrong_answer):
            return i + 1
    for i, m in enumerate(messages):  # 라벨이 없으면 stage=피드백
        if m["role"] == "assistant" and m.get("stage") == "피드백":
            return i
    return None


def early_answer(dialogue: dict) -> bool:
    msgs = dialogue.get("messages", [])
    prob = dialogue.get("problem") or {}
    idx = first_feedback_index(msgs, prob.get("wrong_answer"))
    if idx is None:
        return False
    # 하네스: 아이 턴에 expected_answer 가 있으면 그 문제의 답을 쓴다 (선생님이 낸 문제가 setting 과 다를 수 있음)
    answer = msgs[idx - 1].get("expected_answer") or prob.get("answer")
    return contains_answer(msgs[idx]["content"], answer)


# ---------------------------------------------------------------------------
# 안전 (safety_ok)
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def load_safety_lists(path: str | None = None) -> dict[str, list[str]]:
    p = Path(path) if path else RES / "safety_lists.txt"
    sec: dict[str, list[str]] = {}
    cur = None
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("[") and line.endswith("]"):
            cur = line[1:-1]
            sec[cur] = []
        elif cur:
            sec[cur].append(line)
    return sec


def safety_issues(dialogue: dict) -> list[str]:
    lists = load_safety_lists()
    slang, prof = lists.get("slang", []), lists.get("profanity", [])
    pii = [re.compile(p) for p in lists.get("pii_question", [])]
    issues: list[str] = []
    msgs = dialogue.get("messages", [])
    for i, m in enumerate(msgs):
        if m["role"] != "assistant":
            continue
        t = _strip_think(m["content"])
        prev_child = msgs[i - 1]["content"] if i > 0 and msgs[i - 1]["role"] == "user" else ""
        for w in prof:
            if w in t:
                issues.append(f"turn{i}:profanity:{w}")
        for w in slang:
            if w in t and (w in prev_child or w.startswith("ㅋ") or w.startswith("ㅎ")):
                issues.append(f"turn{i}:slang_echo:{w}")
        for s in sentences(t):
            if s.rstrip().endswith("?") and any(p.search(s) for p in pii):
                issues.append(f"turn{i}:pii_question")
                break
    return issues


def safety_ok(dialogue: dict, safety_fn: Callable[[dict], bool] | None = None) -> bool:
    """목록 대조 + (선택) AI 판정. safety_fn(dialogue) -> True 면 안전."""
    if safety_issues(dialogue):
        return False
    return bool(safety_fn(dialogue)) if safety_fn else True


# ---------------------------------------------------------------------------
# 교정 여부
# ---------------------------------------------------------------------------
def corrected(dialogue: dict) -> bool | None:
    pc = dialogue.get("post_check") or {}
    if "answer" in pc and "student_answer" in pc:
        return _same_answer(pc["student_answer"], pc["answer"])
    if "corrected" in dialogue:
        return bool(dialogue["corrected"])
    return None


def _same_answer(a: str, b: str) -> bool:
    na, nb = norm_answer(a), norm_answer(b)
    if na == nb:
        return True
    try:  # 2/4 == 1/2 도 맞은 것으로
        return _to_frac(na) == _to_frac(nb)
    except (ValueError, ZeroDivisionError):
        return False


# ---------------------------------------------------------------------------
# 오개념 언급 (Base 측정용 / 실패 유형 분류용)
# ---------------------------------------------------------------------------
def mentions_misconception(text: str, keywords: Iterable[str]) -> bool:
    t = _strip_think(text).replace(" ", "")
    return any(k.replace(" ", "") in t for k in keywords if k)


# ---------------------------------------------------------------------------
# 수업 흐름 (기획서: 도입 → 개념 → 확인 질문 → 틀리면 다시 묻기(피드백) → 연습 → 정리)
# stage 라벨이 있는 데이터에서만 쓴다. filters 칸에는 넣지 않고 데이터 검사·보고용.
# ---------------------------------------------------------------------------
STAGE_ORDER = ["도입", "개념", "확인질문", "피드백", "연습", "정리"]


def stage_flow(dialogue: dict) -> dict:
    """{"labeled": 라벨 있음, "ok": 순서가 맞음, "missing": 빠진 단계, "backward": 거꾸로 간 곳}

    피드백 ↔ 확인질문/개념 은 오갈 수 있다 (틀리면 다시 묻기). 도입은 맨 앞, 정리는 맨 끝, 연습은 정리 앞.
    """
    stages = [m.get("stage") for m in dialogue.get("messages", []) if m["role"] == "assistant"]
    if not stages or any(s is None for s in stages):
        return {"labeled": False, "ok": None, "missing": [], "backward": []}
    missing = [s for s in ("도입", "확인질문", "연습", "정리") if s not in stages]
    back = []
    if stages[0] != "도입":
        back.append("도입이 처음이 아님")
    if stages[-1] != "정리":
        back.append("정리가 마지막이 아님")
    if "도입" in stages[1:]:
        back.append("도입이 중간에 다시 나옴")
    if "연습" in stages and "정리" in stages and stages.index("정리") < len(stages) - 1:
        back.append("정리 뒤에 다른 단계")
    # 피드백이 개념·확인질문보다 먼저 와도 허용한다 (첫 문제에서 바로 틀린 경우)
    return {"labeled": True, "ok": not missing and not back, "missing": missing, "backward": back}


# ---------------------------------------------------------------------------
# 대화 1건 -> filters
# ---------------------------------------------------------------------------
def compute_filters(dialogue: dict,
                    judge_fn: Callable[[dict], float | None] | None = None,
                    safety_fn: Callable[[dict], bool] | None = None,
                    thresholds: dict | None = None,
                    return_details: bool = False) -> dict:
    th = thresholds_for(dialogue.get("grade_band"), thresholds)
    tt = teacher_turns(dialogue)
    grade = dialogue.get("grade_band")
    cok, calc_details = calc_ok(dialogue, return_details=True)
    ew, oov = easy_words(tt, grade, return_oov=True)
    f = {
        "corrected": corrected(dialogue),
        "calc_ok": cok,
        "easy_words": ew,
        "avg_sentence_len": avg_sentence_len(tt),
        "question_ratio": question_ratio(tt),
        "early_answer": early_answer(dialogue),
        "safety_ok": safety_ok(dialogue, safety_fn),
        "judge": round(judge_fn(dialogue), 3) if judge_fn else None,
        "duplicate": bool((dialogue.get("filters") or {}).get("duplicate", False)),
    }
    f["passed"] = passes(f, th)
    if return_details:
        f["_details"] = {"calc": calc_details, "oov": oov, "safety": safety_issues(dialogue)}
    return f


def passes(f: dict, th: dict | None = None, grade_band: str | None = None) -> bool:
    th = thresholds_for(grade_band, th)
    ok = (
        f.get("corrected") is not False
        and f["calc_ok"]
        and f["easy_words"] >= th["easy_words_min"]
        and f["avg_sentence_len"] <= th["avg_sentence_len_max"]
        and f["question_ratio"] >= th["question_ratio_min"]
        and not f["early_answer"]
        and f["safety_ok"]
        and not f.get("duplicate", False)
    )
    if f.get("judge") is not None:
        ok = ok and f["judge"] >= th["judge_min"]
    return bool(ok)


# ---------------------------------------------------------------------------
# 다음 선생님 말 1개 (② Base 측정)
# ---------------------------------------------------------------------------
def turn_metrics(scenario: dict, response: str) -> dict:
    """scenario: messages(마지막은 아이 턴) + problem + diag_keywords. response: 다음 선생님 말."""
    msgs = scenario["messages"]
    last_child = msgs[-1]
    fake = {**scenario, "messages": msgs + [{"role": "assistant", "content": response}]}
    answer = last_child.get("expected_answer") or (scenario.get("problem") or {}).get("answer")
    slang_used = bool(last_child.get("response_type") in ("유행어", "비속어"))
    issues = safety_issues({"messages": [last_child, {"role": "assistant", "content": response}]})
    ew, oov = easy_words([response], scenario.get("grade_band"), return_oov=True)
    calc = check_calculations(response, child_said_values(fake))
    return {
        "early_answer": contains_answer(response, answer),
        "easy_words": ew,
        "oov": oov,
        "avg_sentence_len": avg_sentence_len([response]),
        "n_chars": len(_strip_think(response)),
        "mentions_misconception": mentions_misconception(response, scenario.get("diag_keywords", [])),
        "ends_with_question": ends_with_question(response),
        "slang_scene": slang_used,
        "slang_misreaction": slang_used and bool(issues),
        "safety_ok": not issues,
        "calc_ok": all(c["ok"] for c in calc),
    }


# ---------------------------------------------------------------------------
# 데이터셋 단위: 중복 표시
# ---------------------------------------------------------------------------
def _shingles(text: str, k: int = 5) -> set[str]:
    t = re.sub(r"\s+", "", text)
    return {t[i:i + k] for i in range(max(1, len(t) - k + 1))}


def mark_duplicates(dialogues: list[dict], threshold: float = 0.8) -> int:
    """선생님 말 5-gram Jaccard >= threshold 이면 뒤에 나온 것을 duplicate=True. 같은 standard 안에서만 비교.

    반환: duplicate 로 표시한 개수. 각 대화의 filters.duplicate 를 바꾸고 passed 도 다시 계산한다.
    """
    by_std: dict[str, list[tuple[int, set[str], str]]] = {}
    n = 0
    for idx, d in enumerate(dialogues):
        text = " ".join(teacher_turns(d))
        h = hashlib.md5(re.sub(r"\s+", "", text).encode()).hexdigest()
        sh = _shingles(text)
        bucket = by_std.setdefault(d.get("standard", ""), [])
        dup = False
        for _, other, oh in bucket:
            if oh == h or (len(sh | other) and len(sh & other) / len(sh | other) >= threshold):
                dup = True
                break
        f = d.setdefault("filters", {})
        f["duplicate"] = dup
        if "calc_ok" in f:
            f["passed"] = passes(f, grade_band=d.get("grade_band"))
        if dup:
            n += 1
        else:
            bucket.append((idx, sh, h))
    return n


# ---------------------------------------------------------------------------
# GRPO 보상 (⑦ 선택) — 교정 +, 계산 오류·어려운 낱말·바로 정답·긴 말 −
# ---------------------------------------------------------------------------
def grpo_reward(scenario: dict, response: str, corrected_signal: float = 0.0) -> float:
    m = turn_metrics(scenario, response)
    r = 1.0 * corrected_signal
    r += 0.5 if m["mentions_misconception"] else 0.0
    r += 0.3 if m["ends_with_question"] else 0.0
    r -= 1.0 if not m["calc_ok"] else 0.0
    r -= 1.0 if m["early_answer"] else 0.0
    r -= 1.0 if not m["safety_ok"] else 0.0
    r -= 2.0 * max(0.0, THRESHOLDS["easy_words_min"] - m["easy_words"])
    r -= 0.01 * max(0, m["n_chars"] - 120)
    return round(r, 4)


__all__ = [
    "THRESHOLDS", "THRESHOLDS_BY_BAND", "thresholds_for", "stage_flow", "STAGE_ORDER", "compute_filters", "passes", "turn_metrics", "mark_duplicates", "calc_ok", "check_calculations",
    "easy_words", "avg_sentence_len", "question_ratio", "early_answer", "safety_ok", "safety_issues", "corrected",
    "contains_answer", "mentions_misconception", "grpo_reward", "teacher_turns", "sentences",
]

if __name__ == "__main__":  # 간단 사용 예: python -m tutor.metrics data/dialogues.jsonl
    import json
    import sys

    for line in open(sys.argv[1], encoding="utf-8"):
        if line.strip():
            d = json.loads(line)
            print(d.get("id"), json.dumps(compute_filters(d), ensure_ascii=False))
