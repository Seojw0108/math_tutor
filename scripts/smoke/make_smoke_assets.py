"""CPU 점검용 자산 만들기 (실험 결과와 무관, 코드가 끝까지 도는지만 확인).

  python scripts/smoke/make_smoke_assets.py --tokenizer models/Qwen3-4B --out smoke
  -> smoke/tiny-qwen3/  (진짜 Qwen3 토크나이저 + 작은 무작위 Qwen3: 2층, hidden 64)
     smoke/data/misconceptions.jsonl, dialogues.jsonl, dpo.jsonl, sample50.jsonl  (형식만 맞춘 가짜 데이터)

가짜 데이터는 데이터_생성_역할.pdf 형식을 그대로 따른다. A의 실제 데이터가 오면 이 폴더는 지운다.
"""
import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tutor.prompts import LESSON_START, build_system  # noqa: E402
from tutor.students import RULES, solve  # noqa: E402

MCS = [
    ("뺄셈_자리마다큰수빼기", "[4수01-03]", "세 자리 수의 뺄셈", "3~4", "위아래 상관없이 각 자리에서 큰 수 − 작은 수", "50−23", "33", "위 수가 작으면 앞자리에서 10을 빌려 온다"),
    ("뺄셈_빌려준자리안줄임", "[4수01-03]", "세 자리 수의 뺄셈", "3~4", "10을 빌려 오지만 빌려 준 자리는 그대로 둔다", "50−23", "37", "빌려 준 자리는 1 작아진다"),
    ("뺄셈_0에서받아내림", "[4수01-03]", "세 자리 수의 뺄셈", "3~4", "0에서 빌릴 때 0을 9로만 바꾸고 그 앞자리는 줄이지 않는다", "302−145", "257", "0은 빌려 줄 게 없어서 앞자리에서 먼저 빌려 와야 한다"),
    ("나눗셈_나머지가더큼", "[4수01-06]", "나머지가 있는 나눗셈", "3~4", "몫을 작게 잡고 남은 것을 그대로 나머지로 쓴다", "17÷5", "2…7", "나머지가 나누는 수보다 크면 한 묶음 더 만들 수 있다"),
    ("나눗셈_몫나머지바꿈", "[4수01-06]", "나머지가 있는 나눗셈", "3~4", "몫 자리에 나머지, 나머지 자리에 몫을 쓴다", "17÷5", "2…3", "몫 = 몇 묶음, 나머지 = 묶고 남은 개수"),
    ("분수덧셈_분모도더함", "[4수01-15]", "분모가 같은 분수의 덧셈", "3~4", "분자는 분자끼리, 분모는 분모끼리 더한다", "1/4+1/4", "2/8", "조각 크기(분모)는 그대로다"),
    ("분수덧셈_통분없이위아래더함", "[6수01-08]", "분모가 다른 분수의 덧셈", "5~6", "분모가 달라도 위끼리, 아래끼리 더한다", "1/2+1/3", "2/5", "조각 크기가 다르면 먼저 같은 크기로 맞춘다"),
    ("분수덧셈_분자만더함", "[6수01-08]", "분모가 다른 분수의 덧셈", "5~6", "분자만 더하고 분모는 둘 중 큰 것을 쓴다", "1/2+1/3", "2/3", "조각 크기가 다르면 먼저 같은 크기로 맞춘다"),
]


def problem_for(kind, rng, test=False):
    if kind == "sub":
        a = rng.randint(50, 99) if not test else rng.randint(300, 999)
        b = rng.randint(11, a - 10)
        return (a, b), f"{a} − {b}"
    if kind == "div":
        b = rng.randint(3, 9)
        a = b * rng.randint(2, 9) + rng.randint(1, b - 1)
        return (a, b), f"{a} ÷ {b}"
    d = rng.choice([4, 5, 7, 8]) if not test else rng.choice([9, 10, 11])
    a, c = rng.randint(1, d // 2), rng.randint(1, d // 2)
    return ((a, d), (c, d)), f"{a}/{d} + {c}/{d}"


def dialogue(i, mc, rng, split):
    mid, std, sname, grade, rule, _, _, fix = mc
    kind, fn = RULES[mid]
    prob, q = problem_for(kind, rng, split == "test")
    ans, wrong = solve(kind, prob), solve(kind, prob, fn)
    pprob, pq = problem_for(kind, rng, split == "test")
    slang = rng.random() < 0.3
    msgs = [{"role": "system", "content": build_system(grade, sname)}, {"role": "user", "content": LESSON_START},
            {"role": "assistant", "content": f"오늘은 {sname}을 해 볼까요? {q} 는 얼마일까요?", "stage": "도입", "move": "Focus"},
            {"role": "user", "content": f"{wrong}요" + (" ㅋㅋ" if slang else ""), "response_type": "오개념오답"},
            {"role": "assistant", "content": f"그렇게 생각했구나. {fix.split('(')[0]}. 그럼 다시 볼까요?", "stage": "피드백", "move": "Probing"},
            {"role": "user", "content": "음 모르겠어요", "response_type": "모름"},
            {"role": "assistant", "content": "괜찮아요. 하나씩 같이 해 봐요. 처음 자리부터 볼까요?", "stage": "개념", "move": "Focus"},
            {"role": "user", "content": f"{ans}요", "response_type": "정답"},
            {"role": "assistant", "content": f"맞아요! 그럼 {pq} 는 얼마일까요?", "stage": "연습", "move": "Focus"},
            {"role": "user", "content": f"{solve(kind, pprob)}요", "response_type": "정답"},
            {"role": "assistant", "content": "최고예요! 오늘 배운 걸 기억해요. 다음에 또 해 볼까요?", "stage": "정리", "move": "Generic"}]
    passed = rng.random() < 0.7
    return {"id": f"{std.strip('[]')}_{i:04d}", "standard": std, "standard_name": sname, "area": "수와 연산",
            "grade_band": grade, "misconception_id": mid,
            "student_persona": {"style": rng.choice(["짧게 대답", "수다스러움", "소심함", "장난스러움"]), "slang": slang},
            "problem": {"question": q, "answer": ans, "wrong_answer": wrong, "steps": [], "context": "", "template_id": "T01"},
            "post_check": {"question": pq, "answer": solve(kind, pprob), "student_answer": solve(kind, pprob)},
            "messages": msgs, "corrected": True, "split": split,
            "filters": {"passed": passed}, "gen": {"teacher_model": "dummy", "prompt_version": "v1", "seed": i}}


ap = argparse.ArgumentParser()
ap.add_argument("--tokenizer", required=True)
ap.add_argument("--out", default="smoke")
a = ap.parse_args()
out = Path(a.out)
(out / "data").mkdir(parents=True, exist_ok=True)
rng = random.Random(0)

with open(out / "data/misconceptions.jsonl", "w", encoding="utf-8") as f:
    for mid, std, sname, grade, rule, eq, wrong, fix in MCS:
        f.write(json.dumps({"misconception_id": mid, "standard": std, "area": "수와 연산", "grade_band": grade,
                            "name": mid.split("_")[1], "rule": rule, "example": {"question": eq, "wrong_answer": wrong},
                            "fix_point": fix, "simulator": "code"}, ensure_ascii=False) + "\n")
ds = [dialogue(i, MCS[i % len(MCS)], rng, "test" if i % 5 == 0 else "train") for i in range(80)]
with open(out / "data/dialogues.jsonl", "w", encoding="utf-8") as f:
    for d in ds:
        f.write(json.dumps(d, ensure_ascii=False) + "\n")
with open(out / "data/sample50.jsonl", "w", encoding="utf-8") as f:
    for d in ds[:50]:
        f.write(json.dumps({k: v for k, v in d.items() if k not in ("split", "filters")}, ensure_ascii=False) + "\n")
with open(out / "data/dpo.jsonl", "w", encoding="utf-8") as f:
    for d in ds:
        if d["split"] != "train":
            continue
        prompt = [{"role": m["role"], "content": m["content"]} for m in d["messages"][:4]]
        for j, (rej, rt) in enumerate([(f"틀렸어요. 정답은 {d['problem']['answer']}예요.", "바로정답"),
                                       ("분모의 동치 관계를 고려하여 통분 후 연산하십시오.", "어려운용어")]):
            f.write(json.dumps({"id": f"{d['id']}_d{j + 1}", "source_id": d["id"], "cut_after": "오개념오답",
                                "prompt": prompt, "chosen": d["messages"][4]["content"], "rejected": rej,
                                "rejected_type": rt}, ensure_ascii=False) + "\n")

# 작은 무작위 Qwen3 (진짜 토크나이저)
from transformers import AutoTokenizer, Qwen3Config, Qwen3ForCausalLM  # noqa: E402

tok = AutoTokenizer.from_pretrained(a.tokenizer)
cfg = Qwen3Config(vocab_size=len(tok), hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                  num_attention_heads=4, num_key_value_heads=2, head_dim=16, max_position_embeddings=4096,
                  tie_word_embeddings=True, eos_token_id=tok.convert_tokens_to_ids("<|im_end|>"),
                  pad_token_id=tok.pad_token_id, bos_token_id=None)
model = Qwen3ForCausalLM(cfg)
model.save_pretrained(out / "tiny-qwen3")
tok.save_pretrained(out / "tiny-qwen3")
print("smoke assets ->", out, "| dialogues", len(ds), "| params", sum(p.numel() for p in model.parameters()))
