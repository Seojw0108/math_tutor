"""Qwen3 chat 렌더링 + assistant 토큰만 손실 마스킹.

왜 직접 렌더링하나 (① 환경 준비 확인 결과):
  * Qwen3 공식 chat template 은 `{% generation %}` 태그가 없어 TRL `assistant_only_loss=True` 를 쓸 수 없다.
  * 공식 템플릿은 enable_thinking=False 일 때 '마지막' assistant 턴에만 빈 `<think>\\n\\n</think>\\n\\n` 을 붙이고
    그 전 턴들은 think 블록을 지운다. 대화 1건(선생님 턴 7~10개)을 한 시퀀스로 학습하면
    어떤 턴은 think 블록이 있고 어떤 턴은 없는 상태가 되어 추론 때와 모양이 달라진다.
  -> 모든 assistant 턴을 `<|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n{내용}<|im_end|>\\n` 으로 통일한다.
     각 턴의 앞부분은 공식 템플릿의 `add_generation_prompt=True, enable_thinking=False` 출력과 정확히 같다
     (scripts/check_masking.py 가 이것을 검사한다). 학습·DPO·평가 추론 모두 이 함수로 프롬프트를 만든다.

손실: assistant 의 `{내용}<|im_end|>` 토큰만 정답, 나머지(system, user, 'assistant\\n<think>…</think>\\n\\n' 머리)는 -100.
"""
from __future__ import annotations

from typing import Iterable

IM_START = "<|im_start|>"
IM_END = "<|im_end|>"
EMPTY_THINK = "<think>\n\n</think>\n\n"
IGNORE_INDEX = -100


def clean_messages(messages: Iterable[dict]) -> list[dict]:
    """모델 입력에는 role·content 만 (stage/move/response_type 등 메타데이터 제거)."""
    out = []
    for m in messages:
        role, content = m["role"], m["content"]
        if role not in ("system", "user", "assistant"):
            raise ValueError(f"unknown role: {role}")
        out.append({"role": role, "content": content})
    return out


def _block(role: str, content: str) -> str:
    return f"{IM_START}{role}\n{content}{IM_END}\n"


def assistant_header(thinking: bool = False) -> str:
    """생성 프롬프트 꼬리. thinking=False 면 빈 think 블록까지 넣어 바로 답하게 한다."""
    return f"{IM_START}assistant\n" + ("" if thinking else EMPTY_THINK)


def render_prompt(messages: list[dict], thinking: bool = False) -> str:
    """history(마지막이 user) -> 다음 assistant 말을 생성할 프롬프트 문자열.

    thinking=True ("Base + thinking" 비교군)는 공식 템플릿과 같게 이전 assistant 턴에 think 블록을 넣지 않는다.
    """
    msgs = clean_messages(messages)
    parts = []
    for m in msgs:
        if m["role"] == "assistant" and not thinking:
            parts.append(f"{IM_START}assistant\n{EMPTY_THINK}{strip_think(m['content'])}{IM_END}\n")
        elif m["role"] == "assistant":
            parts.append(_block("assistant", strip_think(m["content"])))
        else:
            parts.append(_block(m["role"], m["content"]))
    return "".join(parts) + assistant_header(thinking)


def strip_think(text: str) -> str:
    """생성 결과에서 <think>...</think> 를 떼고 답만 남긴다."""
    if "</think>" in text:
        text = text.split("</think>", 1)[1]
    return text.replace("<think>", "").strip()


def split_think(text: str) -> tuple[str, str]:
    if "</think>" in text:
        th, ans = text.split("</think>", 1)
        return th.replace("<think>", "").strip(), ans.strip()
    return "", text.strip()


def tokenize_for_sft(messages: list[dict], tokenizer, max_length: int = 4096) -> dict:
    """대화 1건 -> input_ids, labels, attention_mask. assistant 내용 + <|im_end|> 만 손실."""
    msgs = clean_messages(messages)
    input_ids: list[int] = []
    labels: list[int] = []

    def add(text: str, train: bool):
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        input_ids.extend(ids)
        labels.extend(ids if train else [IGNORE_INDEX] * len(ids))

    for m in msgs:
        if m["role"] == "assistant":
            add(f"{IM_START}assistant\n{EMPTY_THINK}", train=False)
            add(f"{m['content'].strip()}{IM_END}", train=True)
            add("\n", train=False)
        else:
            add(_block(m["role"], m["content"]), train=False)

    input_ids, labels = input_ids[:max_length], labels[:max_length]
    return {"input_ids": input_ids, "labels": labels, "attention_mask": [1] * len(input_ids)}


def dpo_example(prompt_messages: list[dict], chosen: str, rejected: str) -> dict:
    """dpo.jsonl 한 줄 -> TRL DPOTrainer '표준(문자열)' 형식.

    prompt 는 SFT 와 같은 렌더링 + 생성 머리. chosen/rejected 는 내용만 (EOS=<|im_end|> 는 TRL 이 붙인다).
    """
    if prompt_messages and prompt_messages[-1]["role"] != "user":
        raise ValueError("DPO prompt must end with a user (child) turn")
    return {
        "prompt": render_prompt(prompt_messages, thinking=False),
        "chosen": chosen.strip(),
        "rejected": rejected.strip(),
    }


def transcript(messages: list[dict], include_system: bool = False) -> str:
    """심사·보고서용 사람이 읽는 대화문."""
    names = {"user": "아이", "assistant": "선생님", "system": "설정"}
    lines = []
    for m in messages:
        if m["role"] == "system" and not include_system:
            continue
        lines.append(f"{names[m['role']]}: {strip_think(m['content']) if m['role'] == 'assistant' else m['content']}")
    return "\n".join(lines)
