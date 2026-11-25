"""② Base 측정 결과 -> 실패 패턴 5개 표 + 필터 기준값 제안 + 목록 밖 낱말 (발표 중간 결과용).

  python scripts/base_report.py --gen runs/20261124_base_measure/generations.jsonl
  -> 같은 폴더에 base_report.md, base_report.json, base_report.png

기준값 제안 규칙 (왜 이 값인가):
  학습 데이터는 'Base+프롬프트보다 확실히 나은' 대화만 남겨야 SFT 가 Base 를 넘을 수 있다.
  * easy_words_min       = Base+프롬프트 응답의 75 백분위 (상위 25% 만큼 쉬워야 통과), [0.85, 0.97] 로 자름
  * avg_sentence_len_max = Base+프롬프트 응답의 25 백분위 (짧은 쪽 25% 수준), [12, 30] 로 자름
  * question_ratio_min   = Base+프롬프트 질문 비율 + 0.2, [0.5, 0.8] 로 자름
  * judge_min            = Base+프롬프트 심사 평균 + 0.5, [3.0, 4.2] 로 자름 (심사했을 때만)
  학년군별 값도 같은 규칙으로 따로 계산한다 (학년군마다 말투 기준이 다르다).
"""
import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tutor.io_utils import read_jsonl  # noqa: E402
from tutor.prompts import RUBRIC_KEYS  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--gen", required=True)
ap.add_argument("--ref_mode", default="base_prompt", help="기준값을 잡을 비교군")
a = ap.parse_args()
rows = read_jsonl(a.gen)
out_dir = Path(a.gen).parent


def pct(xs, q):
    xs = sorted(xs)
    if not xs:
        return None
    k = (len(xs) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def mean(xs):
    xs = [x for x in xs if x is not None]
    return round(sum(xs) / len(xs), 4) if xs else None


def clamp(x, lo, hi):
    return None if x is None else round(min(hi, max(lo, x)), 2)


def summarize(rs):
    m = [r["metrics"] for r in rs]
    slang = [x for x in m if x["slang_scene"]]
    judged = [mean([r["judge"][k] for k in RUBRIC_KEYS]) for r in rs if r.get("judge")]
    return {
        "n": len(rs),
        "빈_응답_비율": mean([0.0 if r["response"].strip() else 1.0 for r in rs]),  # thinking 이 길이 안에 안 끝남
        "바로_정답_비율": mean([1.0 if x["early_answer"] else 0.0 for x in m]),
        "쉬운_낱말_비율": mean([x["easy_words"] for x in m]),
        "평균_문장_길이": mean([x["avg_sentence_len"] for x in m]),
        "응답_글자수": mean([x["n_chars"] for x in m]),
        "오개념_언급_비율": mean([1.0 if x["mentions_misconception"] else 0.0 for x in m]),
        "질문으로_끝남_비율": mean([1.0 if x["ends_with_question"] else 0.0 for x in m]),
        "유행어_오반응_비율": mean([1.0 if x["slang_misreaction"] else 0.0 for x in slang]) if slang else None,
        "계산_오류_비율": mean([0.0 if x["calc_ok"] else 1.0 for x in m]),
        "AI_심사": mean(judged) if judged else None,
    }


def propose(rs):
    m = [r["metrics"] for r in rs]
    judged = [mean([r["judge"][k] for k in RUBRIC_KEYS]) for r in rs if r.get("judge")]
    q = mean([1.0 if x["ends_with_question"] else 0.0 for x in m])
    p = {
        "easy_words_min": clamp(pct([x["easy_words"] for x in m], 0.75), 0.85, 0.97),
        "avg_sentence_len_max": clamp(pct([x["avg_sentence_len"] for x in m], 0.25), 12, 30),
        "question_ratio_min": clamp(None if q is None else q + 0.2, 0.5, 0.8),
    }
    if judged:
        p["judge_min"] = clamp(mean(judged) + 0.5, 3.0, 4.2)
    return p


by_mode = defaultdict(list)
for r in rows:
    by_mode[r["mode"]].append(r)
table = {mode: summarize(rs) for mode, rs in by_mode.items()}
ref = by_mode.get(a.ref_mode) or rows
proposal = {"all": propose(ref)}
for band in sorted({r["grade_band"] for r in ref}):
    proposal[band] = propose([r for r in ref if r["grade_band"] == band])

by_area = {mode: {area: summarize([r for r in rs if r["area"] == area]) for area in sorted({r["area"] for r in rs})}
           for mode, rs in by_mode.items()}
oov = Counter(w for r in rows for w in r["metrics"]["oov"])

# ---- 저장 ----
res = {"table": table, "proposal": proposal, "by_area": by_area, "top_oov": oov.most_common(80)}
(out_dir / "base_report.json").write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")

keys = list(next(iter(table.values())).keys())
lines = ["# ② Base 측정 결과", "", f"장면 {len({r['scenario_id'] for r in rows})}개 × 샘플 "
         f"{len({r['sample'] for r in rows})}개", "", "| 지표 | " + " | ".join(table) + " |",
         "|---|" + "---|" * len(table)]
for k in keys:
    lines.append(f"| {k} | " + " | ".join("" if table[mo][k] is None else str(table[mo][k]) for mo in table) + " |")
lines += ["", f"## 필터 기준값 제안 (기준: {a.ref_mode})", "", "| 범위 | " + " | ".join(proposal["all"]) + " |",
          "|---|" + "---|" * len(proposal["all"])]
for band, p in proposal.items():
    lines.append(f"| {band} | " + " | ".join(str(p.get(k)) for k in proposal["all"]) + " |")
lines += ["", "규칙: 쉬운 낱말 = 비교군 75 백분위, 문장 길이 = 25 백분위, 질문 비율 = 비교군 + 0.2, 심사 = 평균 + 0.5.",
          "-> A와 함께 확정한 뒤 tutor/metrics.py 의 THRESHOLDS 에 반영.", "",
          "## 목록 밖 낱말 상위 (초등 어휘 목록 보강 후보)", "",
          ", ".join(f"{w}({c})" for w, c in oov.most_common(60)), "",
          "## 응답 예시 (장면별 첫 샘플)", ""]
for sid in sorted({r["scenario_id"] for r in rows})[:8]:
    lines.append(f"### {sid}")
    for r in rows:
        if r["scenario_id"] == sid and r["sample"] == 0:
            lines.append(f"- **{r['mode']}**: {r['response'][:300]}")
    lines.append("")
(out_dir / "base_report.md").write_text("\n".join(lines), encoding="utf-8")

try:
    from tutor.plotting import setup_korean_font
    plt = setup_korean_font()
    pats = ["바로_정답_비율", "오개념_언급_비율", "질문으로_끝남_비율", "유행어_오반응_비율"]
    fig, ax = plt.subplots(figsize=(8, 4))
    w = 0.8 / len(table)
    for i, mo in enumerate(table):
        ax.bar([j + i * w for j in range(len(pats))], [table[mo][p] or 0 for p in pats], w, label=mo)
    ax.set_xticks([j + 0.4 - w / 2 for j in range(len(pats))], [p.replace("_", " ") for p in pats])
    ax.set_ylim(0, 1)
    ax.legend()
    ax.set_title("Base 실패 패턴")
    fig.tight_layout()
    fig.savefig(out_dir / "base_report.png", dpi=150)
except Exception as e:  # 그래프는 선택
    print("plot skipped:", e)

print("\n".join(lines[:20]))
print("saved", out_dir / "base_report.md")
