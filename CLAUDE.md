# CLAUDE.md — 새 세션 인수인계

Claude Code가 세션을 시작할 때 자동으로 읽는 파일이다. 이전 세션(2026-11-24~26, 개인 Windows PC)에서
정한 것과 아직 남은 것을 적었다. 명령과 원리 설명은 `README.md`에 있으니 먼저 README를 훑는다.

## 이 프로젝트와 사용자
- 팀 프로젝트 **LLM-Math_Tutor**: 초등 수학 오개념을 질문으로 고쳐 주는 선생님 모델.
  Qwen3-4B 위에 LoRA로 SFT → DPO 한다. **모델을 처음부터 만드는 것이 아니다.**
- 사용자는 **역할 B(학습·평가 파이프라인)** 담당이다. 역할 A는 데이터(대화·DPO 쌍) 생성과 코드 학생 담당.
- 기획 문서 3개(`파이프라인_역할.pdf`, `데이터_생성_역할.pdf`, `매쓰 튜터 with SLM.pdf` 중간발표 자료)는
  개인 PC에만 있고 **git에 올리지 않는다**(`.gitignore`의 `*.pdf`). 필요한 내용은 이 파일과 README에 옮겨 두었다.
- 답은 **한국어**로, 쉬운 말로 한다. 커밋·푸시는 사용자가 요청할 때만 한다.
- 저장소: https://github.com/Seojw0108/math_tutor (main)

## 지금 상황 (2026-11-26)
- 첫 버전(커밋 274639b) 푸시 완료. CPU에서 작은 가짜 Qwen3로 전체 흐름을 점검했다
  (base_measure → SFT → DPO → run_eval → make_results, 테스트 14개 통과).
- A의 데이터를 기다리지 않고 **학교 GPU 서버(도커 컨테이너, 리눅스)** 에서 먼저 ①②를 돌리기로 했다.
  - 서버 첫 설정: `bash scripts/server_setup.sh [--download] [--claude]` → 유지 폴더에 `env.sh`, `venv`, `server_check.txt`.
  - 접속할 때마다 `source <유지 폴더>/env.sh`.
  - 도커라서 overlay 영역의 파일은 컨테이너를 다시 만들면 사라질 수 있다. 모델·결과·venv는 유지 폴더에 둔다.
- 일정: 1주차 11/21~27, 2주차 11/28~12/4, 3주차 12/5~12/8(발표). README "일정" 절 참고.

## 서버에서 다음 할 일 (순서대로)
1. `server_setup.sh` 결과(`server_check.txt`) 확인: GPU 이름·VRAM, bf16, 인터넷 막힘 여부.
2. ① `check_masking.py --model models/Qwen3-4B`, `pytest -q`, `check_env.py` 판정 확인.
3. ② `base_measure.py --n_samples 3` → `base_report.py`. 처음 진짜 4B를 돌리는 것이므로
   메모리·속도(문장 생성 시간)를 기록하고 README "규모와 시간 어림"을 실제 값으로 고친다.
4. `base_report.md`의 기준값 제안을 A와 맞춰 `tutor/metrics.py`의 `THRESHOLDS`·`THRESHOLDS_BY_BAND`에 넣는다.
5. 데이터 없이도 할 수 있는 확인: SFT 메모리 시험(가짜/소량 데이터로 `train_sft.py` 몇 스텝),
   `--engine vllm`, QLoRA(bitsandbytes), gradio 데모. 모두 아직 실행해 본 적 없다.
6. 결과를 커밋한다. `runs/*/ckpt/`, `runs/*/adapter/`, `models/`, `data/`는 git 제외다.

## A의 파일이 오면 결정 순서
1. 샘플 50개 / 시범 100개 → `apply_filters.py` 로 filters 채우고 `.stats.json` 확인
   (어느 필터에서 많이 떨어지는지, 목록 밖 낱말 상위 → `vocab_tools.py`로 어휘 목록 보강).
2. 기준값을 A와 확정(너무 많이 떨어지면 기준 완화 vs 데이터 프롬프트 수정 중 결정).
3. 시범 100개로 `train_sft.py` 코드 점검(손실 하강, samples.jsonl 말투) → 본 학습은 split=train & passed=true.
4. SFT 응답을 `export_sft_responses.py`로 A에게 → A가 dpo.jsonl 생성 → `train_dpo.py`.
5. `run_eval.py`로 5개 모드(base, base_prompt, base_thinking, sft, sft_dpo) 같은 시드 평가 → `make_results.py`.

## 데이터 형식 (요약, 원문은 데이터_생성_역할.pdf)
- dialogues.jsonl 한 줄 = 수업 1건:
  `id, standard("[4수01-03]"), standard_name, area, grade_band("1~2"|"3~4"|"5~6"), misconception_id,
  student_persona, problem{question, answer, wrong_answer, context}, messages[{role, content, (stage, response_type ...)}],
  post_check{question, answer}, split("train"|"val"|"test"), filters{...}`
  - messages의 메타데이터 칸은 학습 입력에서 지운다(`tutor/chat.py: clean_messages`).
  - 첫 user 턴은 `[수업 시작]`, system은 `tutor/prompts.py`의 `SYSTEM_TEMPLATE`.
- dpo.jsonl 한 줄: `prompt(messages 목록), chosen, rejected, rejected_type`.
- 평가용 `data/misconceptions.jsonl`(오개념 목록)과 A의 코드 학생 모듈(`run_eval.py --student_module 패키지.모듈:클래스`)도 A에게서 받는다.
- 예시는 `probe/base_probe.jsonl`.

## 고치기 전에 알아야 할 설계 결정 (코드만 봐서는 이유가 안 보이는 것)
- **마스킹을 직접 한다.** Qwen3 템플릿에 `{% generation %}`이 없어 TRL `assistant_only_loss`를 못 쓴다.
  모든 선생님 턴을 `<|im_start|>assistant\n<think>\n\n</think>\n\n{내용}<|im_end|>` 로 렌더링하고
  손실은 내용+`<|im_end|>`에만 건다. SFT·DPO·평가 추론이 **같은 렌더링**(`render_prompt`)을 써야 한다.
- **DPO**(TRL 1.14): 문자열 prompt/chosen/rejected를 받고 EOS는 TRL이 붙인다. base+SFT 병합 모델에
  새 LoRA를 붙이므로 기준 모델 = SFT 모델. `max_length`는 앞쪽을 남기고 자르므로 긴 쌍은 미리 걸러 낸다.
  margin이 안 오르면 lr 5e-6 → 5e-5 시도(configs/dpo.yaml 주석).
- **transformers 5.x**: `warmup_ratio`가 없어져 `runlog.warmup_kwargs`로 처리, `torch_dtype` 대신 `dtype=`.
- **easy_words**는 kiwipiepy 형태소 분석 기준. A도 `kiwipiepy==0.25.1`을 써야 값이 같다.
- **교정 성공률**: 사후 문제가 오개념으로도 맞힐 수 있는 문제면(`post_check_diagnostic=False`) 계산에서 뺀다.
  안 빼면 엉터리 선생님도 성공률이 나왔다.
- **AI 심사**는 6항목 루브릭, 둘 비교는 순서를 바꿔 두 번 묻는다. API는 환경 변수 `JUDGE_*`/`STUDENT_*`
  (openai 호환 / anthropic / dummy), 응답은 `.cache/llm`에 저장.
- `RuleStudent`는 A의 코드 학생 대신 쓰는 흉내다. 발표용 평가는 A의 모듈(`--student_module`)로 한다.
- 성취기준은 코드가 121개(고시 원문), 중간발표 자료는 125개(출판사본). 발표에서 하나로 맞춰야 한다.
- 실험 결과는 `eval.tag`로 구분한다(`run_eval.py --tag`). 같은 모드 여러 실험이 섞이지 않게.
