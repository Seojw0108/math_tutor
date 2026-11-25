"""평가 하네스 (⑤).

설정(성취기준·학년군·오개념·성격) -> 수업(평가할 모델 ↔ 학생, 최대 12턴) -> 사후 문제 -> 채점(교정 여부, 턴 수, ③ 지표, ④ 점수)

여러 수업을 '같은 턴끼리 묶어' 배치 생성한다 (300회 × 12턴 을 한 개씩 돌리면 너무 느리다).
"""
from __future__ import annotations

import copy
from typing import Callable

from . import metrics as M
from .prompts import LESSON_START, build_system

SETTING_KEYS = ["id", "standard", "standard_name", "area", "grade_band", "misconception_id",
                "student_persona", "problem", "post_check", "split"]


def setting_from_dialogue(d: dict) -> dict:
    s = {k: copy.deepcopy(d.get(k)) for k in SETTING_KEYS if k in d}
    if "post_check" in s:
        s["post_check"] = {k: v for k, v in s["post_check"].items() if k != "student_answer"}
    return s


def run_lessons(teacher, settings: list[dict], student_factory: Callable[[dict], object],
                max_turns: int = 12, seed: int = 0, model_name: str = "model",
                student_name: str = "code", judge=None, log: Callable[[str], None] = print) -> list[dict]:
    states = []
    for i, s in enumerate(settings):
        st = student_factory(s)
        st.reset(s, seed=seed + i)
        msgs = [{"role": "system", "content": build_system(s["grade_band"], s["standard_name"])},
                {"role": "user", "content": LESSON_START}]
        states.append({"setting": s, "student": st, "messages": msgs, "done": False,
                       "understood_turn": None, "solved_after_understood": False})

    for turn in range(1, max_turns + 1):
        active = [x for x in states if not x["done"]]
        if not active:
            break
        outs = teacher.generate([x["messages"] for x in active], seed=seed * 1000 + turn)
        for x, o in zip(active, outs):
            x["messages"].append({"role": "assistant", "content": o.text,
                                  **({"thinking": o.thinking} if o.thinking else {})})
            reply = x["student"].respond(x["messages"])
            um = {"role": "user", "content": reply["content"]}
            if reply.get("response_type"):
                um["response_type"] = reply["response_type"]
            if reply.get("expected_answer"):
                um["expected_answer"] = reply["expected_answer"]
            x["messages"].append(um)
            if reply.get("understood") and x["understood_turn"] is None:
                x["understood_turn"] = turn
            # 알겠다 + 이후 문제를 맞힘 = 교정 시연 -> 수업 끝
            if x["understood_turn"] is not None and reply.get("response_type") == "정답" and reply.get("expected_answer"):
                x["solved_after_understood"] = True
                x["done"] = True
        log(f"  turn {turn}: active {len(active)}")

    results = []
    for x in states:
        s, st = x["setting"], x["student"]
        d = {**copy.deepcopy(s), "messages": x["messages"]}
        pc = s.get("post_check") or {}
        diag = None
        if pc.get("question"):
            d["post_check"] = {**pc, "student_answer": st.answer_post_check(pc["question"])}
            if hasattr(st, "is_diagnostic"):
                diag = st.is_diagnostic(pc["question"])
        n_teacher = sum(1 for m in x["messages"] if m["role"] == "assistant")
        f = M.compute_filters(d, judge_fn=judge.dialogue_score if judge else None, return_details=True)
        details = f.pop("_details")
        d["filters"] = f
        d["eval"] = {
            "model": model_name, "student": student_name, "teacher_turns": n_teacher,
            "understood_turn": x["understood_turn"], "solved_after_understood": x["solved_after_understood"],
            "corrected": bool(f["corrected"]),
            "post_check_diagnostic": diag,  # False = 오개념이어도 맞히는 문제 -> 교정 성공률에서 뺀다
            "showed_misconception": any(m.get("response_type") in M.WRONG_TYPES for m in x["messages"]
                                        if m["role"] == "user"),
            "calc_errors": [c for c in details["calc"] if not c["ok"]],
            "safety_issues": details["safety"],
            "oov": details["oov"],
        }
        d["eval"]["failure_type"] = classify_failure(d)
        results.append(d)
    return results


def classify_failure(d: dict) -> str | None:
    """교정 실패 수업 유형 (⑨). 우선순위 순서로 첫 번째 해당 유형."""
    ev, f = d["eval"], d["filters"]
    if ev.get("post_check_diagnostic") is False:
        return "사후문제_판별불가"
    if ev["corrected"]:
        return None
    if not ev["showed_misconception"]:
        return "문제_제시_없음"      # 오개념이 드러날 문제를 내지 않음
    if f["early_answer"]:
        return "바로_정답"
    if not f["calc_ok"]:
        return "계산_오류"
    if not f["safety_ok"]:
        return "안전_말투"
    if ev["understood_turn"] is None:
        return "진단_실패"           # 오개념 핵심을 짚지 못함
    return "확인_실패"               # 알겠다 했지만 사후 문제 틀림


def summarize(results: list[dict]) -> dict:
    """결과 표 한 줄: 교정 성공률, 교정 턴 수, 질문 비율, 계산 오류율, 어려운 낱말, AI 심사."""
    n = len(results)
    if not n:
        return {}
    valid = [r for r in results if r["eval"].get("post_check_diagnostic") is not False]
    corr = [r for r in valid if r["eval"]["corrected"]]
    turns = [r["eval"]["understood_turn"] for r in corr if r["eval"]["understood_turn"]]
    judged = [r["filters"]["judge"] for r in results if r["filters"].get("judge") is not None]
    mean = lambda xs: round(sum(xs) / len(xs), 4) if xs else None  # noqa: E731
    return {
        "n": n,
        "correction_rate": mean([1.0 if r["eval"]["corrected"] else 0.0 for r in valid]),
        "n_non_diagnostic": n - len(valid),
        "turns_to_correct": mean(turns),
        "question_ratio": mean([r["filters"]["question_ratio"] for r in results]),
        "calc_error_rate": mean([0.0 if r["filters"]["calc_ok"] else 1.0 for r in results]),
        "hard_word_ratio": mean([1 - r["filters"]["easy_words"] for r in results]),
        "avg_sentence_len": mean([r["filters"]["avg_sentence_len"] for r in results]),
        "early_answer_rate": mean([1.0 if r["filters"]["early_answer"] else 0.0 for r in results]),
        "safety_ok_rate": mean([1.0 if r["filters"]["safety_ok"] else 0.0 for r in results]),
        "judge": mean(judged),
    }
