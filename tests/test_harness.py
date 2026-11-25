from tutor.generation import GenOutput
from tutor.harness import run_lessons, summarize
from tutor.students import RuleStudent, solve, RULES

SETTING = {
    "id": "t1", "standard": "[4수01-15]", "standard_name": "분모가 같은 분수의 덧셈", "area": "수와 연산",
    "grade_band": "3~4", "misconception_id": "분수덧셈_분모도더함",
    "student_persona": {"style": "짧게 대답", "slang": False},
    "problem": {"question": "1/4 + 1/4", "answer": "2/4", "wrong_answer": "2/8"},
    "post_check": {"question": "1/5 + 2/5", "answer": "3/5"},
}

GOOD = ["피자 한 판을 4조각으로 나눴어요. 1/4 + 1/4 는 얼마일까요?",
        "위 숫자도 아래 숫자도 더했구나. 피자 조각 크기가 바뀌었어요?",
        "맞아요, 조각 크기는 그대로예요. 그럼 아래 숫자는 그대로일까요?",
        "그럼 2/7 + 3/7 은 얼마일까요?",
        "최고예요!"]
BAD = ["1/4 + 1/4 는 얼마일까요?", "틀렸어요. 정답은 2/4예요. 다음 문제 3/8 + 2/8 은요?",
       "정답은 5/8 이에요.", "잘했어요.", "끝!"]


class Scripted:
    def __init__(self, lines):
        self.lines = lines

    def generate(self, batch, seed=0):
        return [GenOutput(text=self.lines[min(sum(m["role"] == "assistant" for m in msgs), len(self.lines) - 1)])
                for msgs in batch]


def test_rules_match_pdf_examples():
    assert solve("sub", (50, 23), RULES["뺄셈_자리마다큰수빼기"][1]) == "33"
    assert solve("sub", (50, 23), RULES["뺄셈_빌려준자리안줄임"][1]) == "37"
    assert solve("sub", (302, 145), RULES["뺄셈_0에서받아내림"][1]) == "257"
    assert solve("div", (17, 5), RULES["나눗셈_나머지가더큼"][1]) == "2…7"
    assert solve("div", (17, 5), RULES["나눗셈_몫나머지바꿈"][1]) == "2…3"
    assert solve("fadd", ((1, 4), (1, 4)), RULES["분수덧셈_분모도더함"][1]) == "2/8"
    assert solve("fadd", ((1, 2), (1, 3)), RULES["분수덧셈_통분없이위아래더함"][1]) == "2/5"
    assert solve("fadd", ((1, 2), (1, 3)), RULES["분수덧셈_분자만더함"][1]) == "2/3"
    assert solve("fadd", ((1, 4), (1, 4))) == "2/4"
    assert solve("sub", (50, 23)) == "27" and solve("div", (17, 5)) == "3…2"


def test_good_teacher_corrects_bad_teacher_does_not():
    good = run_lessons(Scripted(GOOD), [SETTING], lambda s: RuleStudent(), max_turns=6, log=lambda m: None)[0]
    bad = run_lessons(Scripted(BAD), [SETTING], lambda s: RuleStudent(), max_turns=6, log=lambda m: None)[0]
    assert good["eval"]["corrected"] and good["eval"]["understood_turn"] is not None, good["messages"]
    assert good["filters"]["early_answer"] is False
    assert not bad["eval"]["corrected"]
    assert bad["filters"]["early_answer"] is True
    assert bad["eval"]["failure_type"] == "바로_정답"
    s = summarize([good, bad])
    assert s["correction_rate"] == 0.5 and s["early_answer_rate"] == 0.5
