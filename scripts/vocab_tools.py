"""③ 초등 기본 어휘 목록 관리 (어휘 목록 확보도 B).

  # 데이터에서 자주 나오는 '목록 밖' 낱말 뽑기 -> 사람이 보고 쉬운 말이면 basic_words.txt 에 추가
  python scripts/vocab_tools.py oov --inp data/dialogues.jsonl --top 200 --out results/oov.tsv
  # 외부 어휘 목록(한 줄에 한 낱말 또는 TSV 첫 칸) 합치기. 예: 국립국어원 학습용 어휘 A·B 등급
  python scripts/vocab_tools.py merge --src external_words.txt
"""
import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tutor import metrics as M  # noqa: E402
from tutor.io_utils import read_jsonl  # noqa: E402

ap = argparse.ArgumentParser()
sub = ap.add_subparsers(dest="cmd", required=True)
o = sub.add_parser("oov")
o.add_argument("--inp", required=True)
o.add_argument("--top", type=int, default=200)
o.add_argument("--out", default="results/oov.tsv")
mg = sub.add_parser("merge")
mg.add_argument("--src", required=True)
a = ap.parse_args()

if a.cmd == "oov":
    cnt, by_band = Counter(), {}
    for d in read_jsonl(a.inp):
        _, oov = M.easy_words(M.teacher_turns(d), d.get("grade_band"), return_oov=True)
        cnt.update(oov)
        for w in oov:
            by_band.setdefault(w, Counter())[d.get("grade_band")] += 1
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        f.write("word\tcount\tby_grade\tadd?(y/n)\n")
        for w, c in cnt.most_common(a.top):
            f.write(f"{w}\t{c}\t{dict(by_band[w])}\t\n")
    print(f"top {a.top} OOV -> {a.out}")
else:
    basic = set(M.load_basic_words())
    new = []
    for line in open(a.src, encoding="utf-8"):
        w = line.split("\t")[0].strip()
        if w and not w.startswith("#") and w not in basic:
            new.append(w)
            basic.add(w)
    path = M.RES / "basic_words.txt"
    with open(path, "a", encoding="utf-8") as f:
        f.write(f"\n# ---- merged from {Path(a.src).name} ({len(new)}) ----\n")
        for i in range(0, len(new), 20):
            f.write(" ".join(new[i:i + 20]) + "\n")
    print(f"added {len(new)} words -> {path}")
