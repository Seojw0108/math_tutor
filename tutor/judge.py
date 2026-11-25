"""AI 심사 (④).

  * 루브릭 6항목(공감·격려, 구체물·비유, 단계 나눔, 오개념 대응, 눈높이, 안전·말투) 각 1~5점 -> 평균 = filters.judge
  * 데이터 생성 모델과 다른 계열로 심사 (LLMClient 설정)
  * 비교 채점은 순서를 바꿔 2번, 모델 이름 숨김 (compare)
  * 50건 사람 채점과 일치도 -> scripts/judge_agreement.py

metrics.compute_filters(d, judge_fn=Judge().dialogue_score) 처럼 쓴다.
"""
from __future__ import annotations

from statistics import mean

from .chat import transcript
from .llm_client import LLMClient
from .prompts import (JUDGE_SYSTEM, RUBRIC_KEYS, judge_dialogue_prompt, judge_pair_prompt, judge_safety_prompt,
                      judge_turn_prompt)


def _clean_scores(obj: dict) -> dict:
    out = {}
    for k in RUBRIC_KEYS:
        v = obj.get(k)
        try:
            v = int(round(float(v)))
        except (TypeError, ValueError):
            continue
        out[k] = min(5, max(1, v))
    if len(out) != len(RUBRIC_KEYS):
        raise ValueError(f"missing rubric keys: {obj}")
    out["reason"] = str(obj.get("reason", ""))[:300]
    return out


class Judge:
    def __init__(self, client: LLMClient | None = None, misconceptions: dict | None = None):
        self.client = client or LLMClient(prefix="JUDGE")
        self.misconceptions = misconceptions or {}

    def _hint(self, d: dict) -> str | None:
        mc = self.misconceptions.get(d.get("misconception_id", ""))
        if mc:
            return f"{mc.get('name', '')} (아이 규칙: {mc.get('rule', '')}; 짚을 점: {mc.get('fix_point', '')})"
        return d.get("misconception_id")

    # ---- 대화 1건 ----
    def dialogue_scores(self, d: dict) -> dict:
        p = judge_dialogue_prompt(d.get("grade_band", ""), d.get("standard_name", ""),
                                  transcript(d["messages"]), self._hint(d))
        return _clean_scores(self.client.chat_json(JUDGE_SYSTEM, p))

    def dialogue_score(self, d: dict) -> float:
        s = self.dialogue_scores(d)
        return round(mean(s[k] for k in RUBRIC_KEYS), 3)

    # ---- 다음 선생님 말 1개 ----
    def turn_scores(self, d: dict, context_messages: list[dict], response: str) -> dict:
        p = judge_turn_prompt(d.get("grade_band", ""), d.get("standard_name", ""),
                              transcript(context_messages), response, self._hint(d))
        return _clean_scores(self.client.chat_json(JUDGE_SYSTEM, p))

    # ---- 비교 채점: 순서 바꿔 2번 ----
    def compare(self, d: dict, context_messages: list[dict], resp_a: str, resp_b: str) -> dict:
        ctx = transcript(context_messages)
        g, s = d.get("grade_band", ""), d.get("standard_name", "")
        r1 = self.client.chat_json(JUDGE_SYSTEM, judge_pair_prompt(g, s, ctx, resp_a, resp_b)).get("winner")
        r2 = self.client.chat_json(JUDGE_SYSTEM, judge_pair_prompt(g, s, ctx, resp_b, resp_a)).get("winner")
        # r1: 1=A, 2=B / r2: 1=B, 2=A
        a1 = {1: "A", 2: "B"}.get(int(r1 or 0), "tie")
        a2 = {1: "B", 2: "A"}.get(int(r2 or 0), "tie")
        winner = a1 if a1 == a2 else "tie"
        return {"winner": winner, "first": a1, "swapped": a2, "consistent": a1 == a2}

    # ---- 안전 AI 판정 (metrics.safety_ok 의 safety_fn) ----
    def is_safe(self, d: dict) -> bool:
        obj = self.client.chat_json(JUDGE_SYSTEM, judge_safety_prompt(transcript(d["messages"])))
        return not bool(obj.get("unsafe", False))
