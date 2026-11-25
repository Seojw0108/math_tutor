import copy

from tutor import metrics as M

SAMPLE = {
    "id": "4수01-15_0137", "standard": "[4수01-15]", "standard_name": "분모가 같은 분수의 덧셈",
    "area": "수와 연산", "grade_band": "3~4", "misconception_id": "분수덧셈_분모도더함",
    "student_persona": {"style": "짧게 대답", "slang": True},
    "problem": {"question": "1/4 + 1/4", "answer": "2/4", "wrong_answer": "2/8",
                "steps": ["분모가 같은지 본다", "분자끼리 더한다", "분모는 그대로 둔다"],
                "context": "피자", "template_id": "T03"},
    "post_check": {"question": "1/5 + 2/5", "answer": "3/5", "student_answer": "3/5"},
    "messages": [
        {"role": "system", "content": "너는 초등학교 선생님이야. 학년군: 3~4. 성취기준: 분모가 같은 분수의 덧셈. 쉬운 말로, 한 번에 하나씩 질문하며 가르쳐."},
        {"role": "user", "content": "[수업 시작]"},
        {"role": "assistant", "content": "피자 한 판을 4조각으로 나눴어요. 그중 1조각은 분수로 어떻게 쓸까요?", "stage": "도입", "move": "Focus"},
        {"role": "user", "content": "1/4요", "response_type": "정답"},
        {"role": "assistant", "content": "딩동댕! 그럼 1/4조각에 1/4조각을 더 먹으면 얼마나 먹은 걸까요?", "stage": "확인질문", "move": "Focus"},
        {"role": "user", "content": "2/8요 ㅋㅋ", "response_type": "오개념오답"},
        {"role": "assistant", "content": "위 숫자도 아래 숫자도 더했구나. 피자 조각 크기가 바뀌었어요, 그대로예요?", "stage": "피드백", "move": "Probing"},
        {"role": "user", "content": "그대로요", "response_type": "정답"},
        {"role": "assistant", "content": "맞아요. 조각 크기는 그대로니까 아래 숫자 4도 그대로예요. 그럼 조각은 몇 개일까요?", "stage": "개념", "move": "Probing"},
        {"role": "user", "content": "2개요", "response_type": "정답"},
        {"role": "assistant", "content": "좋아요! 그래서 1/4 + 1/4 = 2/4 예요. 1/5 + 2/5 는 얼마일까요?", "stage": "연습", "move": "Focus"},
        {"role": "user", "content": "3/5요", "response_type": "정답"},
        {"role": "assistant", "content": "최고예요! 오늘은 조각 크기는 그대로, 조각 수만 더한다는 걸 배웠어요. 다음에 또 해 볼까요?", "stage": "정리", "move": "Generic"},
    ],
    "corrected": True, "split": "train",
}


def test_sample_passes_basic_filters():
    f = M.compute_filters(SAMPLE, return_details=True)
    assert f["corrected"] is True
    assert f["calc_ok"] is True, f["_details"]["calc"]
    assert f["early_answer"] is False
    assert f["safety_ok"] is True, f["_details"]["safety"]
    assert f["question_ratio"] == 1.0
    assert 0 < f["avg_sentence_len"] < 30
    assert f["easy_words"] >= 0.8, f["_details"]["oov"]
    assert f["judge"] is None


def test_calc_errors_detected():
    assert M.check_calculations("50−23=27 이에요")[0]["ok"]
    assert not M.check_calculations("50 - 23 = 33 이에요")[0]["ok"]
    assert M.check_calculations("1/2 + 1/3 = 5/6")[0]["ok"]
    assert not M.check_calculations("1/2+1/3=2/5")[0]["ok"]
    assert M.check_calculations("17÷5=3…2")[0]["ok"]
    assert not M.check_calculations("17 ÷ 5 = 2…7")[0]["ok"]
    assert M.check_calculations("2 + 3 × 4 = 14")[0]["ok"]
    assert M.check_calculations("1/4 + 1/4 = 1/2")[0]["ok"]
    assert M.check_calculations("3 × 4 = 12")[0]["ok"]
    assert M.check_calculations("0.5 + 0.25 = 0.75")[0]["ok"]
    # 아이 오답을 되짚는 식은 무시
    assert M.check_calculations("1/4+1/4=2/8 이라고 했구나", {"2/8"}) == []


def test_calc_ok_dialogue_level():
    d = copy.deepcopy(SAMPLE)
    d["messages"][10]["content"] = "그래서 1/4 + 1/4 = 2/5 예요."
    assert M.calc_ok(d) is False


def test_early_answer():
    d = copy.deepcopy(SAMPLE)
    d["messages"][6]["content"] = "틀렸어요. 정답은 2/4예요. 분모가 같으면 분자만 더해요."
    assert M.early_answer(d) is True
    assert not M.contains_answer("12/45 를 봐요", "2/4")
    assert M.contains_answer("17÷5 는 3 나머지 2 예요", "3…2")


def test_safety():
    d = copy.deepcopy(SAMPLE)
    d["messages"][6]["content"] = "ㅋㅋ 2/8이라니! 다시 볼까요?"
    assert M.safety_ok(d) is False
    d = copy.deepcopy(SAMPLE)
    d["messages"][2]["content"] = "안녕! 이름이 뭐예요? 어디 살아요?"
    assert M.safety_ok(d) is False
    d = copy.deepcopy(SAMPLE)
    d["messages"][5]["content"] = "개쉽네 2/8"
    d["messages"][6]["content"] = "개쉽다고? 다시 보자"
    assert M.safety_ok(d) is False


def test_question_ratio_and_sentences():
    assert M.ends_with_question("그대로예요? 😊")
    assert M.ends_with_question("어떻게 될까요?!")
    assert not M.ends_with_question("잘했어요!")
    assert len(M.sentences("하나예요. 둘이에요? 셋!")) == 3
    assert len(M.sentences("0.5 는 반이에요.")) == 1


def test_duplicates_and_passed():
    a, b = copy.deepcopy(SAMPLE), copy.deepcopy(SAMPLE)
    b["id"] = "x"
    for d in (a, b):
        d["filters"] = M.compute_filters(d)
    assert M.mark_duplicates([a, b]) == 1
    assert b["filters"]["duplicate"] and not b["filters"]["passed"]
    assert not a["filters"]["duplicate"]


def test_passes_judge_threshold():
    f = M.compute_filters(SAMPLE, judge_fn=lambda d: 3.0)
    assert f["judge"] == 3.0 and f["passed"] is False


def test_turn_metrics():
    sc = {"grade_band": "3~4", "problem": {"answer": "2/4", "wrong_answer": "2/8"},
          "diag_keywords": ["조각 크기", "분모는 그대로", "아래 숫자"],
          "messages": SAMPLE["messages"][:6]}
    good = M.turn_metrics(sc, "위 숫자도 아래 숫자도 더했구나. 조각 크기가 바뀌었어요?")
    bad = M.turn_metrics(sc, "ㅋㅋ 틀렸어요. 정답은 2/4입니다. 통분의 원리에 따라 분자만 더합니다.")
    assert good["mentions_misconception"] and good["ends_with_question"] and not good["early_answer"]
    assert bad["early_answer"] and not bad["ends_with_question"] and not bad["safety_ok"]
    assert M.grpo_reward(sc, "위 숫자도 아래 숫자도 더했구나. 조각 크기가 바뀌었어요?") > M.grpo_reward(sc, "정답은 2/4입니다.")


def test_slide3_generic_ai_vs_tutor():
    """중간발표 자료 3쪽: 일반 AI 응답은 바로 정답 + 어려운 말, 우리 튜터 응답은 질문으로 짚기."""
    sc = {"grade_band": "3~4", "problem": {"answer": "2/4", "wrong_answer": "2/8"},
          "diag_keywords": ["조각 크기", "아래 숫자", "위 숫자도 아래 숫자도"],
          "messages": SAMPLE["messages"][:6]}
    generic = "틀렸습니다. 분모가 같은 분수는 분자끼리만 더합니다. 정답은 2/4이고, 약분하면 1/2입니다."
    tutor = "위 숫자도 아래 숫자도 더했구나! 피자를 떠올려 볼까요? 조각 크기가 바뀌었어요, 그대로예요?"
    g, t = M.turn_metrics(sc, generic), M.turn_metrics(sc, tutor)
    assert g["early_answer"] and not t["early_answer"]
    assert "약분" in g["oov"]                  # 약분은 5~6 용어 -> 3~4 에서는 어려운 말
    assert t["easy_words"] > g["easy_words"]
    assert t["ends_with_question"] and not g["ends_with_question"]
    assert t["mentions_misconception"]


def test_stage_flow():
    assert M.stage_flow(SAMPLE)["ok"] is True
    bad = copy.deepcopy(SAMPLE)
    bad["messages"][-1]["stage"] = "연습"
    f = M.stage_flow(bad)
    assert f["ok"] is False and "정리" in f["missing"]
    assert M.stage_flow({"messages": [{"role": "assistant", "content": "x"}]})["labeled"] is False


def test_grade_band_thresholds():
    assert M.thresholds_for("1~2")["avg_sentence_len_max"] < M.thresholds_for("5~6")["avg_sentence_len_max"]
    f = {"corrected": True, "calc_ok": True, "easy_words": 0.95, "avg_sentence_len": 22.0, "question_ratio": 0.8,
         "early_answer": False, "safety_ok": True, "judge": None, "duplicate": False}
    assert M.passes(f, grade_band="3~4") and not M.passes(f, grade_band="1~2")
