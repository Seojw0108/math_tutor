#!/usr/bin/env bash
# ⑧ 추가 실험 일괄 실행 (엘리스 GPU). 최소 2개: 필터 전후 ★, 데이터 크기 ★. 시간이 남으면 LoRA rank, QLoRA.
# 각 실험 = SFT 학습 -> 코드 학생 평가 (같은 test 설정·같은 시드).
#   bash scripts/run_experiments.sh filter size        # ★ 두 개
#   bash scripts/run_experiments.sh rank qlora         # 선택
set -euo pipefail
N_EVAL=${N_EVAL:-300}
MODEL=${MODEL:-models/Qwen3-4B}

latest() { ls -d runs/*_"$1" | sort | tail -1; }
train_eval() {  # $1=name, 나머지=--set 인자
  local name=$1; shift
  python scripts/train_sft.py --config configs/sft.yaml --set name="$name" model_path="$MODEL" "$@"
  local run; run=$(latest "$name")
  python scripts/run_eval.py --model_name sft --model "$MODEL" --sft_adapter "$run/adapter" \
      --student code --n "$N_EVAL" --tag "$name"
}

for exp in "$@"; do
  case $exp in
    filter)  # Raw vs Filtered
      train_eval exp_raw data_mode=raw "why=필터 효과: passed 무시한 train 전체"
      train_eval exp_filtered data_mode=filtered "why=필터 효과: passed=true 만" ;;
    size)    # 25% / 50% / 100% (같은 seed 라 25% ⊂ 50% ⊂ 100%)
      for f in 0.25 0.5 1.0; do train_eval "exp_frac${f/./}" fraction=$f "why=데이터 크기 $f"; done ;;
    rank)
      for r in 8 16 32; do train_eval "exp_r$r" lora.r=$r lora.alpha=$((2*r)) "why=LoRA rank $r (alpha=2r 고정)"; done ;;
    qlora)
      train_eval exp_qlora qlora=true "why=LoRA bf16 vs QLoRA 4bit: 메모리·시간 대비 성능" ;;
    *) echo "unknown experiment $exp"; exit 1 ;;
  esac
done
python scripts/make_results.py --evals "runs/*_eval_sft_code_exp_*" --out results/experiments
