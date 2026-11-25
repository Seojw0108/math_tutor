"""A용: dialogues.jsonl 에 metrics.py 필터를 적용해 filters 칸을 채운다 (+ 중복 표시, 통계).

  python scripts/apply_filters.py --inp data/dialogues_raw.jsonl --out data/dialogues.jsonl [--judge] [--safety_ai]
  -> out 파일 + out.stats.json (필터별 탈락 수, passed 비율, 목록 밖 낱말 상위)
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tutor import metrics as M  # noqa: E402
from tutor.io_utils import read_jsonl, write_jsonl  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--inp", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--judge", action="store_true", help="AI 심사 점수 포함 (JUDGE_* 환경 변수)")
ap.add_argument("--safety_ai", action="store_true", help="safety_ok 에 AI 판정도 더함")
ap.add_argument("--dup_threshold", type=float, default=0.8)
a = ap.parse_args()

judge = None
if a.judge or a.safety_ai:
    from tutor.judge import Judge
    judge = Judge()
ds = read_jsonl(a.inp)
oov, fails = Counter(), Counter()
for d in ds:
    f = M.compute_filters(d, judge_fn=judge.dialogue_score if a.judge else None,
                          safety_fn=judge.is_safe if a.safety_ai else None, return_details=True)
    oov.update(f.pop("_details")["oov"])
    d["filters"] = f
n_dup = M.mark_duplicates(ds, a.dup_threshold)
for d in ds:
    f, t = d["filters"], M.thresholds_for(d.get("grade_band"))
    fails["corrected"] += f["corrected"] is False
    fails["calc_ok"] += not f["calc_ok"]
    fails["easy_words"] += f["easy_words"] < t["easy_words_min"]
    fails["avg_sentence_len"] += f["avg_sentence_len"] > t["avg_sentence_len_max"]
    fails["question_ratio"] += f["question_ratio"] < t["question_ratio_min"]
    fails["early_answer"] += bool(f["early_answer"])
    fails["safety_ok"] += not f["safety_ok"]
    fails["judge"] += f.get("judge") is not None and f["judge"] < t["judge_min"]
    fails["duplicate"] += bool(f["duplicate"])
flows = [M.stage_flow(d) for d in ds]
labeled = [f for f in flows if f["labeled"]]
stage_stats = {"labeled": len(labeled), "order_ok": sum(1 for f in labeled if f["ok"]),
               "missing": dict(Counter(m for f in labeled for m in f["missing"])),
               "backward": dict(Counter(b for f in labeled for b in f["backward"]))}
write_jsonl(a.out, ds)
stats = {"n": len(ds), "passed": sum(d["filters"]["passed"] for d in ds),
         "passed_ratio": round(sum(d["filters"]["passed"] for d in ds) / max(len(ds), 1), 4),
         "fail_counts": dict(fails), "duplicates": n_dup, "thresholds": {b: M.thresholds_for(b) for b in M.GRADE_ORDER},
         "by_split": dict(Counter((d.get("split"), d["filters"]["passed"]) for d in ds).most_common()),
         "stage_flow": stage_stats, "top_oov": oov.most_common(60)}
stats["by_split"] = {f"{k[0]}|passed={k[1]}": v for k, v in stats["by_split"].items()}
Path(a.out + ".stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps({k: v for k, v in stats.items() if k != "top_oov"}, ensure_ascii=False, indent=2))
