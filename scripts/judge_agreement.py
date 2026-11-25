"""④ 50건은 사람이 같은 루브릭으로 채점 -> AI 심사와 일치도 보고.

1) 채점지 만들기:
   python scripts/judge_agreement.py make --dialogues data/dialogues.jsonl --n 50 --out results/human_scoring.csv
   (CSV 를 엑셀로 열어 항목별 1~5점을 채운다. AI 점수는 다른 파일에 따로 저장해 사람이 못 보게 한다)
2) 일치도:
   python scripts/judge_agreement.py score --human results/human_scoring.csv --ai results/human_scoring_ai.json
   -> 항목별 정확 일치율, ±1 일치율, Spearman, quadratic weighted kappa
"""
import argparse
import csv
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tutor.chat import transcript  # noqa: E402
from tutor.io_utils import read_jsonl  # noqa: E402
from tutor.prompts import RUBRIC, RUBRIC_KEYS  # noqa: E402


def rank(xs):
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    r = [0.0] * len(xs)
    i = 0
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        for k in range(i, j + 1):
            r[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return r


def pearson(x, y):
    n = len(x)
    mx, my = sum(x) / n, sum(y) / n
    sx = sum((a - mx) ** 2 for a in x) ** 0.5
    sy = sum((b - my) ** 2 for b in y) ** 0.5
    return None if sx == 0 or sy == 0 else sum((a - mx) * (b - my) for a, b in zip(x, y)) / (sx * sy)


def qwk(x, y, lo=1, hi=5):
    k = hi - lo + 1
    O = [[0] * k for _ in range(k)]
    for a, b in zip(x, y):
        O[a - lo][b - lo] += 1
    n = len(x)
    ha = [sum(O[i]) for i in range(k)]
    hb = [sum(O[i][j] for i in range(k)) for j in range(k)]
    num = sum(((i - j) ** 2 / (k - 1) ** 2) * O[i][j] for i in range(k) for j in range(k))
    den = sum(((i - j) ** 2 / (k - 1) ** 2) * ha[i] * hb[j] / n for i in range(k) for j in range(k))
    return None if den == 0 else 1 - num / den


ap = argparse.ArgumentParser()
sub = ap.add_subparsers(dest="cmd", required=True)
m = sub.add_parser("make")
m.add_argument("--dialogues", default="data/dialogues.jsonl")
m.add_argument("--n", type=int, default=50)
m.add_argument("--out", default="results/human_scoring.csv")
m.add_argument("--with_ai", action="store_true", help="AI 심사도 돌려 *_ai.json 에 저장 (JUDGE_* 필요)")
s = sub.add_parser("score")
s.add_argument("--human", required=True)
s.add_argument("--ai", required=True)
a = ap.parse_args()

if a.cmd == "make":
    ds = read_jsonl(a.dialogues)
    random.Random(0).shuffle(ds)
    ds = ds[: a.n]
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["id", "grade_band", "standard_name", "dialogue"] + [f"{k}({n})" for k, n, _ in RUBRIC] + ["memo"])
        for d in ds:
            w.writerow([d["id"], d.get("grade_band"), d.get("standard_name"), transcript(d["messages"])]
                       + [""] * len(RUBRIC) + [""])
    guide = Path(a.out).with_suffix(".rubric.md")
    guide.write_text("# 채점 기준 (AI 심사와 같은 루브릭)\n\n" + "\n".join(f"- **{n}** ({k}): {d}" for k, n, d in RUBRIC),
                     encoding="utf-8")
    print("채점지:", a.out, "/ 기준:", guide)
    if a.with_ai:
        from tutor.judge import Judge
        j = Judge()
        ai = {d["id"]: j.dialogue_scores(d) for d in ds}
        p = Path(a.out).with_name(Path(a.out).stem + "_ai.json")
        p.write_text(json.dumps(ai, ensure_ascii=False, indent=2), encoding="utf-8")
        print("AI 점수(사람에게 보여 주지 말 것):", p)
else:
    ai = json.loads(Path(a.ai).read_text(encoding="utf-8"))
    rows = list(csv.DictReader(open(a.human, encoding="utf-8-sig")))
    report = {}
    for key, name, _ in RUBRIC:
        col = f"{key}({name})"
        pairs = [(int(r[col]), int(ai[r["id"]][key])) for r in rows if r.get(col, "").strip() and r["id"] in ai]
        if not pairs:
            continue
        h, x = [p[0] for p in pairs], [p[1] for p in pairs]
        report[name] = {"n": len(pairs), "exact": round(sum(a == b for a, b in pairs) / len(pairs), 3),
                        "within1": round(sum(abs(a - b) <= 1 for a, b in pairs) / len(pairs), 3),
                        "spearman": None if pearson(rank(h), rank(x)) is None else round(pearson(rank(h), rank(x)), 3),
                        "qwk": None if qwk(h, x) is None else round(qwk(h, x), 3)}
    hm = [sum(int(r[f"{k}({n})"]) for k, n, _ in RUBRIC) / 6 for r in rows
          if all(r.get(f"{k}({n})", "").strip() for k, n, _ in RUBRIC) and r["id"] in ai]
    am = [sum(ai[r["id"]][k] for k in RUBRIC_KEYS) / 6 for r in rows
          if all(r.get(f"{k}({n})", "").strip() for k, n, _ in RUBRIC) and r["id"] in ai]
    if len(hm) > 2:
        report["평균점"] = {"n": len(hm), "pearson": round(pearson(hm, am) or 0, 3),
                         "spearman": round(pearson(rank(hm), rank(am)) or 0, 3)}
    print(json.dumps(report, ensure_ascii=False, indent=2))
    Path(a.human).with_suffix(".agreement.json").write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                                            encoding="utf-8")
