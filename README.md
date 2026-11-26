# LLM-Math_Tutor — B 파이프라인 (학습·평가)

`파이프라인_역할.pdf`의 B 담당 일 전체를 코드로 만든 것이다. A(데이터)가 만든 `dialogues.jsonl`·`dpo.jsonl`로
**Qwen3-4B에 LoRA를 붙여 학습(SFT → DPO)** 하고, **Base보다 나아졌는지 증명**한다.
모델을 처음부터 만들지 않는다. 공개된 Qwen3-4B는 그대로 두고 LoRA(전체의 1% 미만)만 학습한다.

> 상태 (2026-11-24): 모든 스크립트는 **CPU에서 작은 무작위 모델 + 가짜 데이터로 끝까지 도는 것까지 확인**했다
> (`pytest` 14개 통과). 진짜 Qwen3-4B 학습·평가 결과는 아직 없다. 엘리스 GPU에서 처음 돌릴 때 확인할 것은 맨 아래 참고.

## 폴더

```
tutor/                 공용 코드
  metrics.py           ③ 지표 함수 (A에게 주는 파일. 필터 = 최종 평가 지표)
  chat.py              ① Qwen3 렌더링 + 선생님 말에만 손실 마스킹
  prompts.py           고정 system 템플릿, ④ AI 심사 루브릭, AI 학생 프롬프트
  judge.py             ④ AI 심사 (6항목, 비교 채점은 순서 바꿔 2번)
  students.py          ⑤ 학생: 코드 학생 대체(RuleStudent) / AI 학생 / 사람
  harness.py           ⑤ 평가 하네스 (수업 → 사후 문제 → 채점, 배치 생성)
  generation.py        평가 대상 5개 모델 불러오기 (Base, Base+프롬프트, Base+thinking, SFT, SFT+DPO)
  runlog.py            runs/{날짜}_{이름}/ 기록 (공고 요구 항목)
  resources/           성취기준 121개, 초등 기본 어휘, 학년군별 수학 용어, 유행어·비속어·개인정보 목록
configs/sft.yaml, dpo.yaml   ⑥⑦ 초기값 + 왜 그 값인지(why)
probe/base_probe.jsonl       ② 손으로 만든 '아이가 틀린 직후' 장면 28개 (scripts/make_base_probe.py 로 생성)
scripts/                     실행 스크립트 (아래 순서)
tests/                       pytest
```

## 할 일 순서와 명령

엘리스 GPU 서버에서 레포 루트 기준으로 실행한다.

### ① 환경 준비
학교 GPU 서버(도커)에서는 유지되는 폴더에 clone한 뒤 `bash scripts/server_setup.sh --download`를 실행하면
저장 공간·GPU·인터넷 점검, `env.sh`, venv, 패키지 설치, 모델 다운로드까지 한 번에 한다(자세한 것은 스크립트 첫 부분 주석).
```bash
pip install -r requirements.txt
python scripts/download_model.py                  # -> models/Qwen3-4B (git 제외)
python scripts/check_env.py                       # GPU 이름, VRAM, bf16 지원, 메모리 어림, 판정
python scripts/check_masking.py --model models/Qwen3-4B
pytest -q
```
`check_masking.py` 결과(토크나이저로 확인 완료):
- Qwen3 템플릿에 `{% generation %}` 태그가 없다. 그래서 TRL `assistant_only_loss`는 **못 쓰고 직접 마스킹**한다(`tutor/chat.py`).
- 공식 템플릿은 `enable_thinking=False`일 때 마지막 선생님 턴에만 빈 `<think></think>`를 붙이고 이전 턴에서는 지운다.
  대화 1건을 한 번에 학습하면 턴마다 모양이 달라진다. 그래서 **모든 선생님 턴에 빈 think 블록을 넣는 렌더링으로 통일**하고,
  학습·DPO·평가 추론 모두 이 렌더링을 쓴다. 각 턴의 생성 프롬프트는 공식 템플릿과 글자 단위로 같다(검사 OK).
- 손실 토큰 = 선생님 말 + `<|im_end|>`뿐이다(검사 OK). 나머지는 -100.

### ② Base 측정 (데이터 없이 바로)
```bash
python scripts/base_measure.py --model models/Qwen3-4B --n_samples 3 [--judge]
python scripts/base_report.py --gen runs/<날짜>_base_measure/generations.jsonl
```
- 장면 28개: 확정 오개념 8개 × 2, 다른 영역 8개, 중간발표 자료의 아이 말투("몰라요", "어쩔", "그냥 답 알려주세요")와 데모 예시 `52−17=45`.
- 실패 패턴 5개(바로 정답, 어려운 낱말·문장 길이, 오개념 언급, 질문으로 끝남, 유행어 오반응)를 Base / Base+프롬프트 / Base+thinking으로 잰다.
- `base_report.md`에는 표, 그래프(`base_report.png`), **필터 기준값 제안(전체 + 학년군별)**, 어휘 목록 보강 후보가 들어간다.
  제안값을 A와 확정한 뒤 `tutor/metrics.py`의 `THRESHOLDS`·`THRESHOLDS_BY_BAND`에 넣는다.

### ③ 지표 함수 `tutor/metrics.py` → A에게 전달
`compute_filters(대화 1건) -> filters` 결과는 `데이터_생성_역할.pdf`의 filters 칸과 같은 키다.

| 필드 | 계산 |
|---|---|
| corrected | 사후 문제(post_check)를 맞혔나 |
| calc_ok | 선생님 말 속 `a ○ b = c` 식(분수·사칙연산·나머지 나눗셈·혼합 계산)을 검산. 아이 오답을 되짚는 식은 제외 |
| easy_words | 내용어(형태소 분석, kiwipiepy) 중 초등 기본 어휘 + **학년군에 맞는 수학 용어** 비율. 예: '약분'은 5~6 용어라 3~4에선 어려운 말 |
| avg_sentence_len | 선생님 말 평균 문장 길이(글자 수) |
| question_ratio / early_answer | 물음표로 끝나는 턴 비율 / 첫 피드백 턴에 정답이 나오나 |
| safety_ok | 아이 유행어·비속어 따라 하기, 비속어, 개인정보 질문(목록 대조) + 선택 AI 판정 |
| judge | ④ AI 심사 평균 (judge_fn 을 줄 때) |
| duplicate / passed | 같은 성취기준 안 5-gram 유사도 0.8 이상이면 중복 / 모든 기준 통과 |

- A의 사용법: `python scripts/apply_filters.py --inp raw.jsonl --out dialogues.jsonl [--judge]`. 탈락 사유 통계와 **수업 흐름 검사**(도입→개념→확인 질문→피드백→연습→정리, stage 라벨 기준)를 함께 출력한다.
- 기준값은 학년군마다 다르다(1학년은 더 쉽게, 6학년은 조금 더 정확하게). 1~2학년군은 문장 길이 20자 이하·쉬운 낱말 92% 이상, 3~4학년군은 25자·90%, 5~6학년군은 30자·88%. ② 결과로 다시 정한다.
- 어휘 목록: `tutor/resources/basic_words.txt`는 시작용 목록(seed)이다. `python scripts/vocab_tools.py oov --inp data/dialogues.jsonl`로 자주 나오는 목록 밖 낱말을 뽑아 사람이 보고 추가한다. 외부 목록이 있으면 `vocab_tools.py merge`로 합친다.

### ④ AI 심사
- 루브릭 6항목(공감·격려, 구체물·비유, 단계 나눔, 오개념 대응, 눈높이, 안전·말투) 각 1~5점이다. 프롬프트는 `tutor/prompts.py`에 있고 A에게도 준다.
- 심사 모델은 환경 변수로 고른다. **데이터 생성 모델과 다른 계열**로 한다.
  - `JUDGE_PROVIDER=openai|anthropic`
  - `JUDGE_MODEL`, `JUDGE_API_KEY`, (`JUDGE_BASE_URL`)
- 비교 채점은 `Judge.compare`로 하며 순서를 바꿔 2번 부르고 모델 이름은 숨긴다. 응답은 `.cache/llm`에 저장해 같은 채점을 반복하지 않는다.
- 사람 50건과의 일치도:
  ```bash
  python scripts/judge_agreement.py make --n 50 --with_ai     # 채점지 CSV
  python scripts/judge_agreement.py score --human results/human_scoring.csv --ai results/human_scoring_ai.json
  ```
  항목별로 정확 일치율, ±1 일치율, Spearman, QWK를 낸다.

### ⑥ SFT
```bash
# 샘플 50개로 코드 점검 (손실이 내려가고 말투가 선생님이 되는지만, runs/.../samples.jsonl 확인)
python scripts/train_sft.py --set name=sft_check data_path=data/sample50.jsonl data_mode=raw train.epochs=5
# 본 학습: split=train & passed=true
python scripts/train_sft.py
```
- 초기값: LoRA 전체 선형층, r16/α32, lr 2e-4 cosine, warmup 3%, epoch 2, bf16, 최대 3072 토큰.
- `--set key=value`로 바꾸고 이유는 `why`에 적는다.
- `runs/{날짜}_{이름}/`에 config.yaml, log.jsonl(손실), summary.json(학습 방법, LR, batch, epoch, optimizer, rank, trainable params, GPU, 학습 시간, GPU 메모리, why), adapter/가 남는다.

### ⑦ DPO
```bash
python scripts/train_dpo.py --set sft_adapter=runs/<날짜>_sft/adapter
python scripts/export_sft_responses.py --sft_adapter runs/<날짜>_sft/adapter   # A에게: SFT하위 쌍 재료
```
- base에 SFT를 병합한 뒤 새 LoRA를 붙인다. 그래서 기준(reference) 모델은 SFT 모델이 된다. β 0.1, lr 5e-6.
- **주의**: lr 5e-6은 LoRA에선 거의 안 움직일 수 있다. `rewards/margins`가 오르지 않으면 5e-5로 올리고 이유를 기록한다(스크립트가 경고함).
- GRPO(선택)용 보상 함수는 `metrics.grpo_reward`에 있다.

### ⑤ 평가 하네스 (같은 설정·같은 시드, split=test만)
```bash
for m in base base_prompt base_thinking; do python scripts/run_eval.py --model_name $m --student code --n 300; done
python scripts/run_eval.py --model_name sft     --sft_adapter <sft> --student code --n 300
python scripts/run_eval.py --model_name sft_dpo --sft_adapter <sft> --dpo_adapter <dpo> --student code --n 300
python scripts/run_eval.py --model_name sft_dpo ... --student llm --n 200 --judge     # AI 학생, 전 영역
python scripts/run_eval.py --model_name sft_dpo ... --student human --n 20           # 사람
python scripts/math_retention.py --models base,sft,sft_dpo --sft_adapter <sft> --dpo_adapter <dpo>
```
- 수업 흐름: 설정 → 선생님 ↔ 학생(최대 12턴) → 사후 문제 → 채점. 채점 항목은 교정 여부, 교정 턴 수, ③ 지표, ④ 점수다.
- **사후 문제가 오개념을 가려내지 못하면**(오개념대로 풀어도 맞는 숫자) 교정 성공률에서 뺀다. 이런 경우는 `사후문제_판별불가`로 표시한다.
- **코드 학생**은 A의 모듈이 오면 `--student_module 패키지.모듈:클래스`로 꽂는다.
  - 인터페이스는 `tutor/students.py` 맨 위 주석에 있다: `reset`, `respond`, `answer_post_check`, `understood`.
  - 그전까지는 확정 오개념 8개를 흉내 내는 `RuleStudent`가 대신한다.
- `math_retention.py`: 선생님 프롬프트 없이 계산만 물어 **LoRA 뒤에도 수학 실력이 유지되는지** 본다(기획서 모델 원칙).

### ⑧ 추가 실험 (최소 2개)
```bash
bash scripts/run_experiments.sh filter size     # ★ Raw vs Filtered, 데이터 25/50/100% (2,500개면 625/1,250/2,500)
bash scripts/run_experiments.sh rank qlora      # 선택: LoRA rank 8/16/32, bf16 vs 4bit
```

### ⑨ 결과·데모
```bash
python scripts/make_results.py --evals "runs/*_eval_*_code" --out results/code
python scripts/demo.py --sft_adapter <sft> --dpo_adapter <dpo> --compare_base [--share]
python scripts/push_adapter.py --adapter <dpo>/adapter --repo_id <계정>/<이름> --summary <dpo>/summary.json
```
- `make_results.py` 결과물:
  - 맨 위 한 줄: **"교정 성공률 Base+프롬프트 → SFT+DPO (+%p)"**
  - 결과 표(교정 성공률이 첫 열)
  - 영역·학년군별 부록
  - 정확성(계산 오류율) × 눈높이(어려운 낱말) 그래프
  - 교정 실패 유형 분류
  - 같은 설정에서 Base 대 SLM 대화 비교
- 데모: 청중이 아이 역할을 하고 학년군을 바꿀 수 있다. 예시로 `52 빼기 17은 45요!`가 들어 있다. Hugging Face 업로드는 기본이 비공개다.

## A와 주고받기

| 순서 | A → B | B → A |
|---|---|---|
| 1 | 샘플 50개 / 시범 100개 | `tutor/metrics.py` + `scripts/apply_filters.py` + 기준값 제안(`base_report.md`) |
| 2 | 전체 dialogues.jsonl | AI 심사 프롬프트(`tutor/prompts.py`의 RUBRIC, judge_* 함수) |
| 3 | dpo.jsonl | `export_sft_responses.py` 결과(sft_lowest_pairs.jsonl) |

`split` 칸이 없는 샘플·시범 파일도 `train_sft.py`가 모두 train으로 보고 학습한다(코드 점검용).
A도 `kiwipiepy==0.25.1`을 깔아야 easy_words 값이 같다.

## 일정 (중간발표 자료 기준, 오늘 11/24 = 1주차)

| 주 | 팀 전체 | B가 할 것 |
|---|---|---|
| 1주차 11/21~11/27 | 문제 생성기·틀리는 아이, Base 먼저 테스트, 시범 대화 100개 | ① 환경·마스킹 확인 → ② Base 측정 → 기준값 제안 → metrics.py 전달 → 시범 100개로 SFT 코드 점검 |
| 2주차 11/28~12/4 | 대화 2,500개 생성·검사, SFT, 자동 평가 준비 | apply_filters 지원, 본 SFT, 평가 하네스(코드 학생 연결), SFT 응답 → A, ⑧ 실험 ★ 2개 |
| 3주차 12/5~12/8 | DPO, Base vs 우리 모델, 데모, 발표 | DPO, 5개 모델 평가 + 수학 실력 유지, 결과 표·그래프, 데모, HF 업로드, 발표 학습·평가 파트 |

### 규모와 시간 어림 (대화 2,500개 ≈ 선생님 말 2만 5천 개)
- 대화 1건은 약 600~1,000토큰이다. 한 epoch에 약 2백만 토큰, 유효 배치 16이면 약 150스텝이고 epoch 2면 약 300스텝이다.
- SFT LoRA(bf16, grad checkpointing) 어림. 실제 값은 첫 실행의 summary.json으로 바꾼다.
  - A100 80GB: epoch당 약 10~20분
  - 24GB급(A10/L4): epoch당 약 30~60분
- DPO 4,000쌍 × 1 epoch는 약 250스텝이다. 쌍마다 chosen/rejected + 기준 모델을 계산하므로 같은 토큰 수의 SFT보다 2~3배 걸린다.
- 평가: 300수업 × 12턴 × 모델 5개다. HF 배치 생성이면 모델당 수십 분이 걸릴 수 있다. 느리면 `merge_adapters.py`로 병합한 뒤 `--engine vllm`을 쓴다.
- 기획서대로 5,000개로 늘리면 위 시간도 약 2배가 된다.

### 성취기준 수
중간발표 자료는 **125개**(수와 연산 43 / 변화와 관계 13 / 도형과 측정 57 / 자료와 가능성 12)라고 적었다.
`데이터_생성_역할.pdf`는 고시 원문 기준 **121개**(42 / 10 / 57 / 12)이며, 125개는 출판사 배포본 기준이라고 밝힌다.
코드(`tutor/resources/standards.tsv`)는 121개를 쓴다. 발표에서 숫자를 하나로 맞춘다.

## 원리 설명 (팀원 누구든 설명할 수 있게)
- **LoRA**: 원래 가중치 W는 얼리고, 옆에 작은 행렬 두 개(A: d×r, B: r×d, r=16)를 붙여 W + BA로 쓴다. 학습하는 것은 A·B뿐이다.
  - 4B 모델에서 약 3천만 개(1% 미만)만 학습하므로 메모리가 적게 들고, 원래 수학 실력을 덜 망가뜨린다.
  - `math_retention.py`로 실제로 유지되는지 확인한다.
- **SFT**: 좋은 수업 대화를 보여 주고 "다음 선생님 말"을 따라 하게 한다. 손실은 선생님 토큰에만 건다. 아이 말과 system은 정답이 아니다.
- **DPO**: 같은 장면에서 좋은 말(chosen)과 나쁜 말(rejected, 예: 바로 정답)을 주고 좋은 말의 확률을 상대적으로 올린다.
  - β는 SFT 모델에서 얼마나 멀어져도 되는지를 정한다.
  - 보상 모델 없이 쌍만으로 학습한다.
- **평가**: 핵심은 "수업 뒤 아이가 새 문제를 맞혔나"(교정 성공률)다.
  - 아이 역할은 오개념대로 항상 틀리는 코드 학생이고, 선생님이 핵심을 짚어야만 생각을 바꾼다.
  - 같은 설정·같은 시드로 Base와 우리 모델을 비교한다. 말투는 ③ 지표와 ④ AI 심사로 따로 본다.

## 아직 확인 못 한 것 (GPU에서 처음 돌릴 때)
- 진짜 Qwen3-4B에서의 메모리·속도. `check_env.py`로 판정하고, 첫 SFT의 summary.json을 확인한다.
- `--engine vllm` 경로: 코드는 있지만 실행해 보지 않았다.
- `bitsandbytes`(QLoRA), `gradio` 웹 데모: 이 PC에서는 설치·실행하지 않았다. 데모는 터미널 모드(`--cli`)만 확인했다.
- AI 심사·AI 학생의 실제 API 호출: 가짜 응답(`JUDGE_PROVIDER=dummy`)으로만 점검했다.
- `RuleStudent`는 A의 코드 학생을 대신하는 단순한 흉내다. 평가 결과 발표에는 A의 모듈을 쓴다.
