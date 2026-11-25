"""⑨ 데모: 청중이 아이 역할로 수업 받기, 학년군 바꾸기. 옆에 Base 응답을 같이 보여 줄 수 있다.

  python scripts/demo.py --sft_adapter runs/.../sft/adapter --dpo_adapter runs/.../dpo/adapter [--compare_base] [--share]
  python scripts/demo.py --cli ...      # gradio 없이 터미널에서
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tutor.generation import Teacher  # noqa: E402
from tutor.prompts import LESSON_START, build_system  # noqa: E402

PRESETS = {
    "3~4 · 세 자리 수의 뺄셈 (52−17)": ("3~4", "세 자리 수의 뺄셈"),
    "3~4 · 분모가 같은 분수의 덧셈 (1/4+1/4)": ("3~4", "분모가 같은 분수의 덧셈"),
    "3~4 · 나머지가 있는 나눗셈 (17÷5)": ("3~4", "나머지가 있는 나눗셈"),
    "5~6 · 분모가 다른 분수의 덧셈 (1/2+1/3)": ("5~6", "분모가 다른 분수의 덧셈"),
    "1~2 · 받아올림이 있는 덧셈": ("1~2", "두 자리 수의 덧셈"),
}

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="models/Qwen3-4B")
ap.add_argument("--sft_adapter")
ap.add_argument("--dpo_adapter")
ap.add_argument("--compare_base", action="store_true", help="Base+프롬프트 응답도 옆에 (GPU 메모리 2배)")
ap.add_argument("--cli", action="store_true")
ap.add_argument("--share", action="store_true")
a = ap.parse_args()

adapters = [x for x in [a.sft_adapter, a.dpo_adapter] if x]
mode = "sft_dpo" if len(adapters) == 2 else "sft" if adapters else "base_prompt"
tutor = Teacher(model_path=a.model, mode=mode, adapters=adapters, batch_size=1,
                sampling={"temperature": 0.6, "max_new_tokens": 200})
base = Teacher(model_path=a.model, mode="base_prompt", batch_size=1) if a.compare_base else None


def start(grade, standard):
    return [{"role": "system", "content": build_system(grade, standard)}, {"role": "user", "content": LESSON_START}]


def step(history, child=None, who=tutor):
    msgs = history + ([{"role": "user", "content": child}] if child else [])
    out = who.generate([msgs])[0].text
    return msgs + [{"role": "assistant", "content": out}]


if a.cli:
    preset = list(PRESETS)[0]
    grade, std = PRESETS[preset]
    g = input(f"학년군 [{grade}]: ").strip() or grade
    s = input(f"성취기준 [{std}]: ").strip() or std
    h = step(start(g, s))
    print("선생님:", h[-1]["content"])
    while True:
        c = input("아이> ").strip()
        if c in ("/끝", "/quit", ""):
            break
        h = step(h, c)
        print("선생님:", h[-1]["content"])
    sys.exit()

import gradio as gr  # noqa: E402


def to_chat(h):
    return [{"role": m["role"], "content": m["content"]} for m in h if m["role"] != "system"
            and m["content"] != LESSON_START]


def on_start(preset, grade, std):
    if preset in PRESETS and not std.strip():
        grade, std = PRESETS[preset]
    h = step(start(grade, std or PRESETS[preset][1]))
    hb = step(start(grade, std or PRESETS[preset][1]), who=base) if base else []
    return h, hb, to_chat(h), to_chat(hb)


def on_send(child, h, hb):
    if not h:
        return h, hb, [], [], ""
    h = step(h, child)
    hb = step(hb, child, who=base) if base and hb else hb
    return h, hb, to_chat(h), to_chat(hb), ""


def chatbot(label):
    try:
        return gr.Chatbot(label=label, type="messages", height=480)
    except TypeError:  # gradio 6+: messages 형식이 기본
        return gr.Chatbot(label=label, height=480)


with gr.Blocks(title="초등 수학 선생님 SLM") as ui:
    gr.Markdown("## 질문하며 가르치는 초등 수학 선생님\n아이 역할로 일부러 틀려 보세요. 예: `52 빼기 17은 45요!`")
    with gr.Row():
        preset = gr.Dropdown(list(PRESETS), value=list(PRESETS)[0], label="예시 수업")
        grade = gr.Radio(["1~2", "3~4", "5~6"], value="3~4", label="학년군")
        std = gr.Textbox(label="성취기준 (비우면 예시 수업 것)", value="")
        go = gr.Button("수업 시작", variant="primary")
    with gr.Row():
        c1 = chatbot(f"우리 모델 ({mode})")
        c2 = chatbot("Base + 프롬프트") if base else gr.Markdown(visible=False)
    child = gr.Textbox(label="아이 말", placeholder="2/8요 ㅋㅋ")
    h, hb = gr.State([]), gr.State([])
    go.click(on_start, [preset, grade, std], [h, hb, c1, c2] if base else [h, hb, c1, c1])
    child.submit(on_send, [child, h, hb], [h, hb, c1, c2, child] if base else [h, hb, c1, c1, child])
ui.launch(share=a.share)
