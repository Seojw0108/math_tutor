"""⑨ 결과 표·그래프·실패 분류·대화 비교.

  python scripts/make_results.py --evals runs/*_eval_*_code --out results/code
  -> results/code/table.md, table.csv, appendix_area_grade.md, accuracy_vs_level.png,
     failures.md, compare_base_vs_slm.md

결과 표 열: 교정 성공률, 교정 턴 수, 질문 비율, 계산 오류율, 어려운 낱말, AI 심사 (+ 바로 정답, 안전)
"""
import argparse
import csv
import glob
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tutor.chat import transcript  # noqa: E402
from tutor.harness import summarize  # noqa: E402
from tutor.io_utils import read_jsonl  # noqa: E402

ORDER = ["base", "base_prompt", "base_thinking", "sft", "sft_dpo"]
LABEL = {"base": "Base", "base_prompt": "Base+프롬프트", "base_thinking": "Base+thinking", "sft": "SFT",
         "sft_dpo": "SFT+DPO"}
COLS = [("correction_rate", "교정 성공률", "pct"), ("turns_to_correct", "교정 턴 수", "f1"),
        ("question_ratio", "질문 비율", "pct"), ("calc_error_rate", "계산 오류율", "pct"),
        ("hard_word_ratio", "어려운 낱말", "pct"), ("judge", "AI 심사", "f2"),
        ("early_answer_rate", "바로 정답", "pct"), ("safety_ok_rate", "안전 통과", "pct"), ("n", "n", "int")]

ap = argparse.ArgumentParser()
ap.add_argument("--evals", nargs="+", required=True, help="runs/*_eval_* 폴더들 (glob 가능)")
ap.add_argument("--out", default="results")
a = ap.parse_args()
out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)

runs = {}
for pat in a.evals:
    for d in sorted(glob.glob(pat)):
        p = Path(d) / "dialogues.jsonl"
        if p.exists():
            rows = read_jsonl(p)
            if rows:
                key = rows[0]["eval"].get("tag") or rows[0]["eval"]["model"]
                if key in runs:
                    key = Path(d).name
                runs[key] = rows
names = sorted(runs, key=lambda k: (ORDER.index(k) if k in ORDER else 99, k))


def fmt(v, kind):
    if v is None:
        return "-"
    return {"pct": f"{100 * v:.1f}%", "f1": f"{v:.1f}", "f2": f"{v:.2f}", "int": str(v)}[kind]


# ---- 결과 표 ----
summ = {n: summarize(runs[n]) for n in names}
base_k = next((k for k in ["base_prompt", "base"] if k in summ), None)
ours_k = next((k for k in ["sft_dpo", "sft"] if k in summ), None)
headline = ""
if base_k and ours_k and summ[base_k].get("correction_rate") is not None and summ[ours_k].get("correction_rate") is not None:
    b, o = summ[base_k]["correction_rate"], summ[ours_k]["correction_rate"]
    headline = (f"**수업 후 새 문제를 맞힌 비율(교정 성공률): {LABEL[base_k]} {100 * b:.1f}% → {LABEL[ours_k]} "
                f"{100 * o:.1f}% ({100 * (o - b):+.1f}%p)**")
lines = ["# 결과 표", "", headline, "", "| 모델 | " + " | ".join(c[1] for c in COLS) + " |", "|---|" + "---|" * len(COLS)]
for n in names:
    lines.append(f"| {LABEL.get(n, n)} | " + " | ".join(fmt(summ[n].get(k), kind) for k, _, kind in COLS) + " |")
(out / "table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
with open(out / "table.csv", "w", newline="", encoding="utf-8-sig") as f:
    w = csv.writer(f)
    w.writerow(["model"] + [c[0] for c in COLS])
    for n in names:
        w.writerow([n] + [summ[n].get(k) for k, _, _ in COLS])

# ---- 부록: 영역·학년군별 ----
app = ["# 부록: 영역·학년군별 교정 성공률 / 어려운 낱말", ""]
for key, title in [("area", "영역"), ("grade_band", "학년군")]:
    groups = sorted({r.get(key) for n in names for r in runs[n]})
    app += [f"## {title}", "", f"| {title} | " + " | ".join(LABEL.get(n, n) for n in names) + " |",
            "|---|" + "---|" * len(names)]
    for g in groups:
        cells = []
        for n in names:
            s = summarize([r for r in runs[n] if r.get(key) == g])
            cells.append(f"{fmt(s.get('correction_rate'), 'pct')} / {fmt(s.get('hard_word_ratio'), 'pct')}" if s else "-")
        app.append(f"| {g} | " + " | ".join(cells) + " |")
    app.append("")
(out / "appendix_area_grade.md").write_text("\n".join(app), encoding="utf-8")

# ---- 그래프: 정확성(계산 오류율) × 눈높이(어려운 낱말) ----
try:
    from tutor.plotting import setup_korean_font
    plt = setup_korean_font()
    fig, ax = plt.subplots(figsize=(6, 5))
    for n in names:
        s = summ[n]
        x, y = 100 * (s["hard_word_ratio"] or 0), 100 * (s["calc_error_rate"] or 0)
        ax.scatter(x, y, s=60 + 400 * (s["correction_rate"] or 0), alpha=0.7)
        ax.annotate(LABEL.get(n, n), (x, y), textcoords="offset points", xytext=(6, 6))
    ax.set_xlabel("어려운 낱말 비율 (%)  ← 눈높이 좋음")
    ax.set_ylabel("계산 오류율 (%)  ← 정확함")
    ax.set_title("정확성 × 눈높이 (점 크기 = 교정 성공률)")
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(out / "accuracy_vs_level.png", dpi=150)
except Exception as e:
    print("plot skipped:", e)

# ---- 교정 실패 유형 분류 ----
fl = ["# 교정 실패 수업 유형", "", "| 유형 | " + " | ".join(LABEL.get(n, n) for n in names) + " |",
      "|---|" + "---|" * len(names)]
cnt = {n: Counter(r["eval"]["failure_type"] for r in runs[n] if r["eval"]["failure_type"]) for n in names}
for t in sorted({t for c in cnt.values() for t in c}):
    fl.append(f"| {t} | " + " | ".join(str(cnt[n].get(t, 0)) for n in names) + " |")
fl += ["", "유형 정의: 문제_제시_없음(오개념이 드러날 문제를 안 냄) > 바로_정답 > 계산_오류 > 안전_말투 > "
       "진단_실패(핵심을 못 짚음) > 확인_실패(알겠다 했지만 사후 문제 틀림)", ""]
by_mc = defaultdict(Counter)
for n in names:
    for r in runs[n]:
        if not r["eval"]["corrected"]:
            by_mc[r.get("misconception_id")][n] += 1
fl += ["## 오개념별 실패 수", "", "| 오개념 | " + " | ".join(LABEL.get(n, n) for n in names) + " |",
       "|---|" + "---|" * len(names)]
for mc in sorted(by_mc):
    fl.append(f"| {mc} | " + " | ".join(str(by_mc[mc].get(n, 0)) for n in names) + " |")
(out / "failures.md").write_text("\n".join(fl) + "\n", encoding="utf-8")

# ---- 같은 오답(같은 설정)에 대한 Base vs SLM 대화 비교 ----
base_key = next((k for k in ["base_prompt", "base"] if k in runs), None)
slm_key = next((k for k in ["sft_dpo", "sft"] if k in runs), None)
cmp_lines = ["# 같은 설정에 대한 Base vs SLM 대화 비교", ""]
if base_key and slm_key:
    base_by = {r["id"]: r for r in runs[base_key]}
    picked = 0
    for r in runs[slm_key]:  # SLM 은 교정, Base 는 실패한 예를 먼저
        b = base_by.get(r["id"])
        if b and r["eval"]["corrected"] and not b["eval"]["corrected"]:
            cmp_lines += [f"## {r['id']} — {r.get('misconception_id')} ({r.get('grade_band')})", "",
                          f"### {LABEL[base_key]} (교정 실패: {b['eval']['failure_type']})", "```",
                          transcript(b["messages"]), "```", f"### {LABEL[slm_key]} (교정 성공, "
                          f"{r['eval']['understood_turn']}턴)", "```", transcript(r["messages"]), "```", ""]
            picked += 1
            if picked >= 5:
                break
    if not picked:
        cmp_lines.append("(SLM 성공 & Base 실패인 같은 설정이 없음)")
(out / "compare_base_vs_slm.md").write_text("\n".join(cmp_lines), encoding="utf-8")
(out / "summary.json").write_text(json.dumps(summ, ensure_ascii=False, indent=2), encoding="utf-8")
print("\n".join(lines))
print("->", out)
