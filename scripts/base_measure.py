"""② Base 측정: 손으로 만든 '아이가 틀린 직후' 장면에서 Base / Base+프롬프트 / Base+thinking 의 다음 선생님 말 생성.

  python scripts/base_measure.py --model models/Qwen3-4B --n_samples 3 [--judge]
  -> runs/{날짜}_base_measure/generations.jsonl  (이어서 scripts/base_report.py)
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tutor import metrics as M  # noqa: E402
from tutor.chat import clean_messages  # noqa: E402
from tutor.generation import Teacher  # noqa: E402
from tutor.io_utils import read_jsonl, write_jsonl  # noqa: E402
from tutor.runlog import RunLogger  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="models/Qwen3-4B")
ap.add_argument("--scenarios", default="probe/base_probe.jsonl")
ap.add_argument("--modes", default="base,base_prompt,base_thinking")
ap.add_argument("--n_samples", type=int, default=3, help="장면당 샘플 수 (시드 0..n-1)")
ap.add_argument("--batch_size", type=int, default=24)
ap.add_argument("--max_new_tokens_thinking", type=int, default=2048)
ap.add_argument("--judge", action="store_true", help="AI 심사도 함께 (JUDGE_* 환경 변수)")
ap.add_argument("--name", default="base_measure")
ap.add_argument("--limit", type=int, default=None)
a = ap.parse_args()

scenarios = read_jsonl(a.scenarios)[: a.limit]
modes = a.modes.split(",")
log = RunLogger(a.name, config=vars(a) | {"why": "기획서 실패 패턴 5개를 Base 에서 재고, 필터 기준값을 제안하기 위함"})
teacher = Teacher(model_path=a.model, mode=modes[0], batch_size=a.batch_size)
judge = None
if a.judge:
    from tutor.judge import Judge
    judge = Judge()

rows = []
for mode in modes:
    teacher.set_mode(mode)
    if mode == "base_thinking":
        teacher.params["max_new_tokens"] = a.max_new_tokens_thinking
    for k in range(a.n_samples):
        outs = teacher.generate([sc["messages"] for sc in scenarios], seed=1000 * k)
        for sc, o in zip(scenarios, outs):
            row = {"scenario_id": sc["id"], "mode": mode, "sample": k, "misconception_id": sc["misconception_id"],
                   "area": sc["area"], "grade_band": sc["grade_band"], "response": o.text,
                   "thinking_chars": len(o.thinking), "metrics": M.turn_metrics(sc, o.text)}
            if judge:
                row["judge"] = judge.turn_scores(sc, clean_messages(sc["messages"]), o.text)
            rows.append(row)
        log.log(mode=mode, sample=k, n=len(outs))
        print(f"{mode} sample {k}: done")

out = log.dir / "generations.jsonl"
write_jsonl(out, rows)
log.finish(n_rows=len(rows))
print("saved", out, "\nnext: python scripts/base_report.py --gen", out)
