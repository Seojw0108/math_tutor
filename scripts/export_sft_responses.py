"""⑦ SFT 모델 응답 -> A에게 (DPO 'SFT하위' 쌍 재료).

dpo.jsonl 의 prompt(아이 턴까지 자른 대화) 마다 SFT 모델로 k개 응답을 만들고 AI 심사로 채점,
가장 낮은 점수 응답을 rejected 로 하는 쌍 후보를 만든다. chosen 은 원래 대화의 다음 선생님 말(dpo.jsonl 의 chosen).

  python scripts/export_sft_responses.py --sft_adapter runs/20261201_sft/adapter --k 4
  -> runs/{날짜}_sft_responses/sft_responses.jsonl (전체 응답+점수), sft_lowest_pairs.jsonl (dpo.jsonl 형식)
"""
import argparse
import sys
from pathlib import Path
from statistics import mean

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tutor import metrics as M  # noqa: E402
from tutor.generation import load_teacher  # noqa: E402
from tutor.io_utils import read_jsonl, write_jsonl  # noqa: E402
from tutor.judge import Judge  # noqa: E402
from tutor.prompts import RUBRIC_KEYS  # noqa: E402
from tutor.runlog import RunLogger  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="models/Qwen3-4B")
ap.add_argument("--sft_adapter", required=True)
ap.add_argument("--dpo", default="data/dpo.jsonl")
ap.add_argument("--dialogues", default="data/dialogues.jsonl", help="학년군·성취기준 정보를 찾기 위함")
ap.add_argument("--k", type=int, default=4)
ap.add_argument("--limit", type=int, default=None)
ap.add_argument("--batch_size", type=int, default=32)
a = ap.parse_args()

log = RunLogger("sft_responses", config=vars(a) | {"why": "DPO 'SFT하위' 쌍: SFT 모델의 가장 약한 응답을 rejected 로"})
by_id = {d["id"]: d for d in read_jsonl(a.dialogues)} if Path(a.dialogues).exists() else {}
seen, prompts = set(), []
for p in read_jsonl(a.dpo):
    key = (p["source_id"], len(p["prompt"]))
    if key not in seen:  # 같은 자른 지점은 한 번만
        seen.add(key)
        prompts.append(p)
prompts = prompts[: a.limit]

teacher = load_teacher("sft", a.model, sft_adapter=a.sft_adapter, batch_size=a.batch_size)
judge = Judge()
all_rows, pairs = [], []
gens = [teacher.generate([p["prompt"] for p in prompts], seed=100 * j) for j in range(a.k)]
for i, p in enumerate(prompts):
    meta = by_id.get(p["source_id"], {})
    ctx = {"grade_band": meta.get("grade_band", ""), "standard_name": meta.get("standard_name", ""),
           "misconception_id": meta.get("misconception_id"), "problem": meta.get("problem", {}),
           "messages": p["prompt"]}
    scored = []
    for j in range(a.k):
        text = gens[j][i].text
        s = judge.turn_scores(ctx, p["prompt"], text)
        tm = M.turn_metrics(ctx, text)
        scored.append({"response": text, "judge": round(mean(s[k] for k in RUBRIC_KEYS), 3), "scores": s,
                       "early_answer": tm["early_answer"], "easy_words": tm["easy_words"]})
    all_rows.append({"source_id": p["source_id"], "cut_after": p.get("cut_after"), "prompt": p["prompt"],
                     "responses": scored})
    low = min(scored, key=lambda x: x["judge"])
    if low["judge"] < 4.0 and low["response"].strip() != p["chosen"].strip():
        pairs.append({"id": f"{p['source_id']}_s{len(p['prompt'])}", "source_id": p["source_id"],
                      "cut_after": p.get("cut_after"), "prompt": p["prompt"], "chosen": p["chosen"],
                      "rejected": low["response"], "rejected_type": "SFT하위", "rejected_judge": low["judge"]})

write_jsonl(log.dir / "sft_responses.jsonl", all_rows)
write_jsonl(log.dir / "sft_lowest_pairs.jsonl", pairs)
log.finish(n_prompts=len(prompts), n_pairs=len(pairs), judge_calls=judge.client.n_calls)
print(f"{len(prompts)} prompts -> {len(pairs)} SFT하위 pairs. A에게 전달: {log.dir}")
