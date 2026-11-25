"""AI 심사·AI 학생이 쓰는 LLM 호출기. 표준 라이브러리(urllib)만 쓴다.

환경 변수
  JUDGE_PROVIDER   openai | anthropic | dummy   (openai = OpenAI 호환 API: OpenAI, Gemini, vLLM 서버, OpenRouter 등)
  JUDGE_MODEL      모델 이름
  JUDGE_BASE_URL   OpenAI 호환 주소 (기본 https://api.openai.com/v1)
  JUDGE_API_KEY    키 (없으면 OPENAI_API_KEY / ANTHROPIC_API_KEY)
학생용은 같은 이름에 STUDENT_ 접두어 (없으면 JUDGE_ 값을 쓴다).

⚠ 심사 모델은 데이터 생성 모델(A의 gen.teacher_model)과 다른 계열로 고른다 (④).
응답은 cache_dir 에 프롬프트 해시로 저장해 같은 채점을 다시 돈 내고 하지 않는다.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path


class LLMClient:
    def __init__(self, provider: str | None = None, model: str | None = None, base_url: str | None = None,
                 api_key: str | None = None, prefix: str = "JUDGE", cache_dir: str | None = ".cache/llm",
                 temperature: float = 0.0, max_tokens: int = 512):
        env = lambda k: os.environ.get(f"{prefix}_{k}") or os.environ.get(f"JUDGE_{k}")  # noqa: E731
        self.provider = provider or env("PROVIDER") or "dummy"
        self.model = model or env("MODEL") or ""
        self.base_url = (base_url or env("BASE_URL") or "https://api.openai.com/v1").rstrip("/")
        self.api_key = api_key or env("API_KEY") or (
            os.environ.get("ANTHROPIC_API_KEY") if self.provider == "anthropic" else os.environ.get("OPENAI_API_KEY"))
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.cache = Path(cache_dir) if cache_dir else None
        if self.cache:
            self.cache.mkdir(parents=True, exist_ok=True)
        self.n_calls = 0

    # ------------------------------------------------------------------
    def chat(self, system: str, user: str, seed: int = 0) -> str:
        key = hashlib.sha256(json.dumps([self.provider, self.model, system, user, self.temperature, seed],
                                        ensure_ascii=False).encode()).hexdigest()
        cp = self.cache / f"{key}.json" if self.cache else None
        if cp and cp.exists():
            return json.loads(cp.read_text(encoding="utf-8"))["text"]
        text = self._call(system, user, seed)
        self.n_calls += 1
        if cp:
            cp.write_text(json.dumps({"text": text}, ensure_ascii=False), encoding="utf-8")
        return text

    def chat_json(self, system: str, user: str, seed: int = 0, retries: int = 2) -> dict:
        last = ""
        for attempt in range(retries + 1):
            last = self.chat(system, user if attempt == 0 else user + "\n(JSON 하나만 출력)", seed + attempt)
            obj = parse_json(last)
            if obj is not None:
                return obj
        raise ValueError(f"JSON parse failed: {last[:200]}")

    # ------------------------------------------------------------------
    def _call(self, system: str, user: str, seed: int) -> str:
        if self.provider == "dummy":
            return _dummy(system, user)
        if self.provider == "openai":
            url = f"{self.base_url}/chat/completions"
            body = {"model": self.model, "temperature": self.temperature, "max_tokens": self.max_tokens, "seed": seed,
                    "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
            headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
            data = _post(url, body, headers)
            return data["choices"][0]["message"]["content"]
        if self.provider == "anthropic":
            url = "https://api.anthropic.com/v1/messages"
            body = {"model": self.model, "max_tokens": self.max_tokens, "temperature": self.temperature,
                    "system": system, "messages": [{"role": "user", "content": user}]}
            headers = {"x-api-key": self.api_key or "", "anthropic-version": "2023-06-01",
                       "content-type": "application/json"}
            data = _post(url, body, headers)
            return "".join(b.get("text", "") for b in data["content"])
        raise ValueError(f"unknown provider {self.provider}")


def _post(url: str, body: dict, headers: dict, retries: int = 5) -> dict:
    req_body = json.dumps(body).encode()
    for i in range(retries):
        try:
            req = urllib.request.Request(url, data=req_body, headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=120) as r:
                return json.loads(r.read().decode())
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503, 529) and i < retries - 1:
                time.sleep(2 ** i * 2)
                continue
            raise RuntimeError(f"{e.code}: {e.read().decode()[:300]}") from e
        except urllib.error.URLError:
            if i < retries - 1:
                time.sleep(2 ** i * 2)
                continue
            raise
    raise RuntimeError("unreachable")


def parse_json(text: str) -> dict | None:
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                return None
    return None


def _dummy(system: str, user: str) -> str:
    """코드 점검용 가짜 응답 (네트워크 없이 파이프라인을 끝까지 돌려 보기 위함). 점수는 의미 없음."""
    h = int(hashlib.md5(user.encode()).hexdigest(), 16)
    if '"winner"' in user:
        return json.dumps({"winner": h % 3, "reason": "dummy"})
    if '"unsafe"' in user:
        return json.dumps({"unsafe": False, "reason": "dummy"})
    if '"answer"' in user and "문제:" in user:
        return json.dumps({"answer": "모르겠어요"})
    if '"content"' in user:
        return json.dumps({"content": "음... 잘 모르겠어요", "response_type": "모름", "understood": False},
                          ensure_ascii=False)
    keys = ["empathy", "concrete", "stepwise", "misconception", "level", "safety"]
    return json.dumps({k: 1 + (h >> (3 * i)) % 5 for i, k in enumerate(keys)} | {"reason": "dummy"})
