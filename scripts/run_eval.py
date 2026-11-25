"""⑤ 평가 하네스 실행. 평가 대상 5개 모델에 같은 설정·같은 시드로 돌린다. 입력은 split=test 설정만.

  # 코드 학생 (수와 연산, 300회): test 설정을 반복해 300회 채움
  python scripts/run_eval.py --model_name base_prompt --student code --n 300
  python scripts/run_eval.py --model_name sft --sft_adapter runs/.../adapter --student code --n 300
  python scripts/run_eval.py --model_name sft_dpo --sft_adapter ... --dpo_adapter ... --student code --n 300
  # AI 학생 (전 영역)
  python scripts/run_eval.py --model_name sft --sft_adapter ... --student llm --n 200
  # 사람 20회
  python scripts/run_eval.py --model_name sft --sft_adapter ... --student human --n 20

  -> runs/{날짜}_eval_{model}_{student}/dialogues.jsonl, summary.json
"""
import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tutor.generation import load_teacher  # noqa: E402
from tutor.harness import run_lessons, setting_from_dialogue, summarize  # noqa: E402
from tutor.io_utils import read_jsonl, write_jsonl  # noqa: E402
from tutor.runlog import RunLogger  # noqa: E402
from tutor.students import RULES, load_student  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--model_name", required=True, choices=["base", "base_prompt", "base_thinking", "sft", "sft_dpo"])
ap.add_argument("--model", default="models/Qwen3-4B")
ap.add_argument("--sft_adapter")
ap.add_argument("--dpo_adapter")
ap.add_argument("--engine", default="hf", choices=["hf", "vllm"])
ap.add_argument("--dialogues", default="data/dialogues.jsonl")
ap.add_argument("--misconceptions", default="data/misconceptions.jsonl")
ap.add_argument("--student", default="code", choices=["code", "rule", "llm", "human"])
ap.add_argument("--student_module", default=None, help="A의 코드 학생: package.module:ClassName")
ap.add_argument("--n", type=int, default=300, help="수업 횟수 (test 설정을 순환)")
ap.add_argument("--max_turns", type=int, default=12)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--batch_size", type=int, default=32)
ap.add_argument("--judge", action="store_true")
ap.add_argument("--tag", default="")
a = ap.parse_args()

mcs = {m["misconception_id"]: m for m in read_jsonl(a.misconceptions)} if Path(a.misconceptions).exists() else {}
tests = [d for d in read_jsonl(a.dialogues) if d.get("split") == "test"]
if a.student in ("code", "rule"):
    # 코드 학생은 수와 연산 (simulator=code) 오개념만
    code_ids = {k for k, m in mcs.items() if m.get("simulator") == "code"} or set(RULES)
    if not a.student_module:
        code_ids &= set(RULES)
    tests = [d for d in tests if d.get("misconception_id") in code_ids]
if not tests:
    sys.exit("평가할 test 설정이 없습니다")
tests = sorted(tests, key=lambda d: d["id"])
settings = [setting_from_dialogue(tests[i % len(tests)]) for i in range(a.n)]
random.Random(a.seed).shuffle(settings)  # 모든 모델이 같은 순서·같은 시드

name = f"eval_{a.model_name}_{a.student}{('_' + a.tag) if a.tag else ''}"
log = RunLogger(name, config=vars(a) | {"n_unique_settings": len(tests)})
teacher = load_teacher(a.model_name, a.model, sft_adapter=a.sft_adapter, dpo_adapter=a.dpo_adapter,
                       engine=a.engine, batch_size=a.batch_size)
judge = None
if a.judge:
    from tutor.judge import Judge
    judge = Judge(misconceptions=mcs)

results = run_lessons(teacher, settings, lambda s: load_student(a.student, mcs, a.student_module),
                      max_turns=a.max_turns, seed=a.seed, model_name=a.model_name, student_name=a.student,
                      judge=judge, log=lambda msg: log.log(msg=msg))
for r in results:
    r["eval"]["tag"] = a.tag
write_jsonl(log.dir / "dialogues.jsonl", results)
summary = summarize(results)
log.finish(**summary, model_name=a.model_name, student=a.student)
print(summary)
print("->", log.dir)
