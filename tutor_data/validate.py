"""데이터 형식 문서(v1) 기준 검사.

    python -m tutor_data.validate dialogues data/dialogues_sample50.jsonl
    python -m tutor_data.validate dpo data/dpo.jsonl --dialogues data/dialogues.jsonl
    python -m tutor_data.validate misconceptions data/misconceptions.jsonl
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter

from . import spec
from .misconceptions import RULES, correct_answer, load_misconceptions, same_value, wrong_answer
from .problems import CONTEXTS, NUMBER_RANGES, TEMPLATES, _frac_kind, is_diagnostic, op_of
from .student import SLANG, parse_problem

ID_RE = re.compile(r"^(\d수\d\d-\d\d)_(\d{4})$")
STD_RE = re.compile(r"^\[\d수\d\d-\d\d\]$")

_MIS = None


def _mis():
    global _MIS
    if _MIS is None:
        _MIS = load_misconceptions()
    return _MIS


def _keys(errs, name, d, keys):
    if not isinstance(d, dict):
        errs.append(f"{name}: 객체가 아님")
        return False
    if list(d.keys()) != keys:
        errs.append(f"{name}: 키가 다름 {list(d.keys())} ≠ {keys}")
        return False
    return True


def _numbers_in_split(op, standard, ops, split) -> bool:
    if op == "sub":
        lo, hi = NUMBER_RANGES["sub"][split]
        return lo <= ops[0] <= hi
    if op == "div":
        lo, hi = NUMBER_RANGES["div"][split]
        return lo <= ops[0] <= hi
    lo, hi = NUMBER_RANGES[_frac_kind(standard)][split]
    return all(lo <= d <= hi for _, d in ops)


def check_misconception(d: dict) -> list[str]:
    errs = []
    keys = ["misconception_id", "standard", "area", "grade_band", "name", "rule", "example",
            "fix_point", "simulator"]
    if not _keys(errs, "misconception", d, keys):
        return errs
    if not STD_RE.match(d["standard"]):
        errs.append(f"standard 형식: {d['standard']}")
    if d["area"] not in spec.AREAS:
        errs.append(f"area: {d['area']}")
    if d["grade_band"] not in spec.GRADE_BANDS:
        errs.append(f"grade_band: {d['grade_band']}")
    if d["simulator"] not in ("code", "llm"):
        errs.append(f"simulator: {d['simulator']}")
    if (d["area"] == "수와 연산") != (d["simulator"] == "code"):
        errs.append("simulator: 수와 연산은 code, 나머지는 llm")
    _keys(errs, "example", d["example"], ["question", "wrong_answer"])
    if d["simulator"] == "code" and d["misconception_id"] in RULES:
        op = op_of(d["misconception_id"])
        ops = parse_problem(op, d["example"]["question"])
        if ops is None or wrong_answer(d["misconception_id"], ops) != d["example"]["wrong_answer"]:
            errs.append("example.wrong_answer 가 코드 규칙과 다름")
    return errs


def check_dialogue(r: dict) -> list[str]:
    errs: list[str] = []
    if not _keys(errs, "dialogue", r, spec.DIALOGUE_KEYS):
        return errs
    m = ID_RE.match(r["id"])
    if not m or f"[{m.group(1)}]" != r["standard"]:
        errs.append(f"id 형식/성취기준 불일치: {r['id']}")
    mis = _mis().get(r["misconception_id"])
    if mis is None:
        return errs + [f"misconceptions.jsonl 에 없는 오개념: {r['misconception_id']}"]
    for k in ("standard", "area", "grade_band"):
        if r[k] != mis[k]:
            errs.append(f"{k} 가 오개념 목록과 다름")
    if r["area"] not in spec.AREAS:
        errs.append("area 허용 값 아님")
    if r["grade_band"] not in spec.GRADE_BANDS:
        errs.append("grade_band 허용 값 아님")
    if spec.STANDARD_NAMES.get(r["standard"]) != r["standard_name"]:
        errs.append("standard_name 불일치")

    if _keys(errs, "student_persona", r["student_persona"], ["style", "slang"]):
        if r["student_persona"]["style"] not in spec.STYLES:
            errs.append("student_persona.style 허용 값 아님")
        if not isinstance(r["student_persona"]["slang"], bool):
            errs.append("student_persona.slang 은 bool")

    mid, op, split = r["misconception_id"], op_of(r["misconception_id"]), r["split"]
    if split not in spec.SPLITS:
        errs.append("split 허용 값 아님")
        split = "train"
    siblings = [k for k, v in _mis().items() if v["standard"] == r["standard"] and k in RULES]
    p = r["problem"]
    if _keys(errs, "problem", p, spec.PROBLEM_KEYS):
        for k in ("question", "answer", "wrong_answer", "context", "template_id"):
            if not isinstance(p[k], str):
                errs.append(f"problem.{k} 는 문자열")
        ops = parse_problem(op, p["question"])
        if ops is None:
            errs.append("problem.question 을 식으로 읽을 수 없음")
        else:
            if p["answer"] != correct_answer(op, ops):
                errs.append("problem.answer 계산 틀림")
            if p["wrong_answer"] != wrong_answer(mid, ops):
                errs.append("problem.wrong_answer 가 오개념 규칙과 다름")
            if not is_diagnostic(mid, ops, siblings):
                errs.append("진단용 문제 아님 (오답이 갈리지 않음)")
            if not _numbers_in_split(op, r["standard"], ops, split):
                errs.append(f"problem 숫자 범위가 {split} 범위 밖")
        if p["context"] not in CONTEXTS[op][split]:
            errs.append(f"context 가 {split} 목록 밖")
        if p["template_id"] not in TEMPLATES[op][split]:
            errs.append(f"template_id 가 {split} 목록 밖")
        if not (isinstance(p["steps"], list) and p["steps"] and all(isinstance(s, str) for s in p["steps"])):
            errs.append("problem.steps 는 문자열 목록")

    pc = r["post_check"]
    if _keys(errs, "post_check", pc, spec.POST_CHECK_KEYS):
        ops = parse_problem(op, pc["question"])
        if ops is None:
            errs.append("post_check.question 을 식으로 읽을 수 없음")
        else:
            if pc["answer"] != correct_answer(op, ops):
                errs.append("post_check.answer 계산 틀림")
            if not _numbers_in_split(op, r["standard"], ops, split):
                errs.append(f"post_check 숫자 범위가 {split} 범위 밖")
            if same_value(wrong_answer(mid, ops), correct_answer(op, ops)):
                errs.append("post_check 에서 오개념이 드러나지 않음")
        if not isinstance(pc["student_answer"], str):
            errs.append("post_check.student_answer 는 문자열")
        elif r["corrected"] != same_value(pc["student_answer"], pc["answer"]):
            errs.append("corrected 가 post_check 와 맞지 않음")
        if p.get("question") == pc["question"]:
            errs.append("post_check 는 새 문제여야 함")
    if not isinstance(r["corrected"], bool):
        errs.append("corrected 는 bool")

    msgs = r["messages"]
    if not msgs or msgs[0] != {"role": "system", "content": spec.system_prompt(r["grade_band"], r["standard_name"])}:
        errs.append("system 프롬프트가 고정 템플릿과 다름")
    if len(msgs) < 2 or msgs[1] != {"role": "user", "content": spec.START}:
        errs.append('첫 user 는 "[수업 시작]" (response_type 없음)')
    teacher = []
    for i, mm in enumerate(msgs[2:], start=2):
        want = "assistant" if i % 2 == 0 else "user"
        if mm.get("role") != want:
            errs.append(f"messages[{i}] role 순서 틀림")
            break
        if not isinstance(mm.get("content"), str) or not mm["content"].strip():
            errs.append(f"messages[{i}] content 비어 있음")
        if want == "assistant":
            if list(mm.keys()) != ["role", "content", "stage", "move"]:
                errs.append(f"messages[{i}] 키: {list(mm.keys())}")
            if mm.get("stage") not in spec.STAGES:
                errs.append(f"messages[{i}] stage 허용 값 아님")
            if mm.get("move") not in spec.MOVES:
                errs.append(f"messages[{i}] move 허용 값 아님")
            teacher.append(mm)
        else:
            if list(mm.keys()) != ["role", "content", "response_type"]:
                errs.append(f"messages[{i}] 키: {list(mm.keys())}")
            if mm.get("response_type") not in spec.RESPONSE_TYPES:
                errs.append(f"messages[{i}] response_type 허용 값 아님")
            other = SLANG["train" if split == "test" else "test"]
            if any(s in mm["content"] for s in other):
                errs.append(f"messages[{i}] 다른 split 의 유행어 사용")
    lo, hi = spec.TEACHER_TURNS
    if not lo <= len(teacher) <= hi:
        errs.append(f"선생님 턴 {len(teacher)}개 (허용 {lo}~{hi})")
    if teacher and teacher[0]["stage"] != "도입":
        errs.append("첫 선생님 턴 stage 는 도입")
    if not teacher or msgs[-1].get("role") != "assistant" or msgs[-1].get("stage") != "정리":
        errs.append('마지막은 선생님의 "정리" 단계')
    if not any(t["stage"] == "연습" for t in teacher):
        errs.append('"연습" 단계 없음 (post_check 근거)')
    if pc.get("question") and not any(t["stage"] == "연습" and pc["question"] in t["content"] for t in teacher):
        errs.append("post_check 문제가 연습 단계 선생님 말에 없음")

    if _keys(errs, "filters", r["filters"], spec.FILTER_KEYS):
        pass
    if _keys(errs, "gen", r["gen"], spec.GEN_KEYS):
        if r["gen"]["student_model"] != ("code" if mis["simulator"] == "code" else r["gen"]["student_model"]):
            errs.append("gen.student_model: code 학생이면 'code'")
    return errs


def check_dialogues_file(path: str) -> int:
    recs = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    bad = 0
    ids = Counter(r.get("id") for r in recs)
    for i, n in ids.items():
        if n > 1:
            print(f"[중복 id] {i} ×{n}")
            bad += 1
    for r in recs:
        errs = check_dialogue(r)
        if errs:
            bad += 1
            print(f"[{r.get('id')}]\n  - " + "\n  - ".join(errs))
    users = [m for r in recs for m in r["messages"][2:] if m["role"] == "user"]
    rt = Counter(m["response_type"] for m in users)
    with_safety = sum(any(m.get("response_type") in spec.SAFETY_TYPES for m in r["messages"]) for r in recs)
    print(f"\n대화 {len(recs)}개, 오류 있는 대화 {bad}개")
    print(f"split: {dict(Counter(r['split'] for r in recs))}")
    print(f"오개념별: {dict(Counter(r['misconception_id'] for r in recs))}")
    print(f"선생님 턴 수: {dict(sorted(Counter(sum(m['role']=='assistant' for m in r['messages']) for r in recs).items()))}")
    print(f"corrected: {dict(Counter(r['corrected'] for r in recs))}")
    print(f"아이 response_type: {dict(rt)}")
    print(f"안전·유행어·비속어 장면이 있는 대화: {with_safety}/{len(recs)} = {with_safety / max(len(recs), 1):.0%}")
    print(f"stage: {dict(Counter(m['stage'] for r in recs for m in r['messages'] if m['role']=='assistant'))}")
    print(f"move: {dict(Counter(m['move'] for r in recs for m in r['messages'] if m['role']=='assistant'))}")
    return bad


def check_dpo_file(path: str, dialogues: str | None) -> int:
    recs = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    src = {}
    if dialogues:
        src = {d["id"]: d for d in (json.loads(l) for l in open(dialogues, encoding="utf-8") if l.strip())}
    bad = 0
    for r in recs:
        errs = []
        if _keys(errs, "dpo", r, spec.DPO_KEYS):
            if not re.fullmatch(re.escape(r["source_id"]) + r"_d\d+", r["id"]):
                errs.append("id 형식은 {source_id}_d{n}")
            if r["cut_after"] not in spec.RESPONSE_TYPES:
                errs.append("cut_after 허용 값 아님")
            if r["rejected_type"] not in spec.REJECTED_TYPES:
                errs.append("rejected_type 허용 값 아님")
            if not all(list(m.keys()) == ["role", "content"] for m in r["prompt"]):
                errs.append("prompt 는 role·content 만")
            if not r["prompt"] or r["prompt"][-1]["role"] != "user":
                errs.append("prompt 는 아이 턴에서 끝나야 함")
            if r["chosen"] == r["rejected"]:
                errs.append("chosen == rejected")
            d = src.get(r["source_id"])
            if src and d is None:
                errs.append("source_id 대화 없음")
            elif d is not None:
                if d["filters"].get("passed") is not True:
                    errs.append("chosen 은 필터 통과 대화에서만")
                n = len(r["prompt"])
                if [{"role": m["role"], "content": m["content"]} for m in d["messages"][:n]] != r["prompt"]:
                    errs.append("prompt 가 원래 대화 앞부분과 다름")
                elif d["messages"][n - 1].get("response_type") != r["cut_after"]:
                    errs.append("cut_after 가 직전 아이 턴 response_type 과 다름")
                elif d["messages"][n]["content"] != r["chosen"]:
                    errs.append("chosen 이 원래 대화의 다음 선생님 말이 아님")
        if errs:
            bad += 1
            print(f"[{r.get('id')}]\n  - " + "\n  - ".join(errs))
    print(f"\nDPO 쌍 {len(recs)}개, 오류 {bad}개")
    print(f"rejected_type: {dict(Counter(r['rejected_type'] for r in recs))}")
    print(f"cut_after: {dict(Counter(r['cut_after'] for r in recs))}")
    return bad


def main(argv=None):
    if sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("kind", choices=["dialogues", "dpo", "misconceptions"])
    ap.add_argument("path")
    ap.add_argument("--dialogues")
    a = ap.parse_args(argv)
    if a.kind == "dialogues":
        bad = check_dialogues_file(a.path)
    elif a.kind == "dpo":
        bad = check_dpo_file(a.path, a.dialogues)
    else:
        bad = 0
        for l in open(a.path, encoding="utf-8"):
            if l.strip():
                d = json.loads(l)
                e = check_misconception(d)
                if e:
                    bad += 1
                    print(f"[{d.get('misconception_id')}] {e}")
        print(f"오류 {bad}개")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
