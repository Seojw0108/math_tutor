"""데이터 형식 문서(v1)에 적힌 허용 값과 고정 문장."""

FORMAT_VERSION = "v1"
PROMPT_VERSION = "v1"

# system 프롬프트 템플릿 (고정). 바꾸면 PROMPT_VERSION 을 올린다.
SYSTEM_TEMPLATE = ("너는 초등학교 선생님이야. 학년군: {grade_band}. 성취기준: {standard_name}. "
                   "쉬운 말로, 한 번에 하나씩 질문하며 가르쳐.")
START = "[수업 시작]"

AREAS = ["수와 연산", "변화와 관계", "도형과 측정", "자료와 가능성"]
GRADE_BANDS = ["1~2", "3~4", "5~6"]
STYLES = ["짧게 대답", "수다스러움", "소심함", "장난스러움"]
STAGES = ["도입", "개념", "확인질문", "피드백", "연습", "정리"]
MOVES = ["Focus", "Probing", "Telling", "Generic"]
RESPONSE_TYPES = ["정답", "오개념오답", "계산실수", "모름", "딴소리", "유행어", "답요구", "비속어", "개인정보"]
SAFETY_TYPES = ["유행어", "비속어", "개인정보"]
SPLITS = ["train", "test"]
FILTER_KEYS = ["corrected", "calc_ok", "easy_words", "avg_sentence_len", "question_ratio",
               "early_answer", "safety_ok", "judge", "duplicate", "passed"]
GEN_KEYS = ["teacher_model", "student_model", "prompt_version", "seed"]
DIALOGUE_KEYS = ["id", "standard", "standard_name", "area", "grade_band", "misconception_id",
                 "student_persona", "problem", "post_check", "messages", "corrected", "split",
                 "filters", "gen"]
PROBLEM_KEYS = ["question", "answer", "wrong_answer", "steps", "context", "template_id"]
POST_CHECK_KEYS = ["question", "answer", "student_answer"]

DPO_KEYS = ["id", "source_id", "cut_after", "prompt", "chosen", "rejected", "rejected_type"]
CUT_AFTER = ["오개념오답", "모름", "답요구", "유행어", "비속어"]  # "주로" 쓰는 지점
REJECTED_TYPES = ["바로정답", "어려운용어", "계산오류", "질문없음", "오개념무시", "말투위반", "안전위반", "SFT하위"]

TEACHER_TURNS = (8, 12)  # 대화 8~12턴 (선생님 턴 약 7~10개)

# 성취기준 짧은 이름 (system 프롬프트의 standard_name).
# [4수01-15] 는 PDF 예시 그대로, 나머지는 같은 방식(다루는 연산만 남김)으로 줄였다.
STANDARD_NAMES = {
    "[4수01-03]": "세 자리 수의 뺄셈",
    "[4수01-06]": "나누는 수가 한 자리 수인 나눗셈",
    "[4수01-15]": "분모가 같은 분수의 덧셈",
    "[6수01-08]": "분모가 다른 분수의 덧셈",
}


def system_prompt(grade_band: str, standard_name: str) -> str:
    return SYSTEM_TEMPLATE.format(grade_band=grade_band, standard_name=standard_name)
