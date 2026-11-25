"""평가 대상 5개 모델을 같은 방식으로 부르는 래퍼.

  base           system 없음, thinking 끔
  base_prompt    고정 system 템플릿, thinking 끔
  base_thinking  고정 system 템플릿, thinking 켬 (생각은 떼고 답만 쓴다)
  sft            base + SFT LoRA, system 템플릿
  sft_dpo        base + SFT LoRA(병합) + DPO LoRA

engine="hf" (기본) 는 transformers 배치 생성, engine="vllm" 은 병합된 모델 폴더가 필요하다
(scripts/merge_adapters.py). 같은 설정·같은 시드로 돌리기 위해 seed 를 배치마다 고정한다.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .chat import render_prompt, split_think

MODES = {
    "base": {"system": False, "thinking": False},
    "base_prompt": {"system": True, "thinking": False},
    "base_thinking": {"system": True, "thinking": True},
    "sft": {"system": True, "thinking": False},
    "sft_dpo": {"system": True, "thinking": False},
}

# Qwen3 권장 샘플링 (non-thinking / thinking)
SAMPLING = {
    False: {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "max_new_tokens": 256},
    True: {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "max_new_tokens": 2048},
}


@dataclass
class GenOutput:
    text: str          # 답 (think 제거)
    thinking: str = ""  # thinking 모드일 때 생각 부분
    raw: str = ""


@dataclass
class Teacher:
    model_path: str
    mode: str = "base_prompt"
    adapters: list[str] = field(default_factory=list)
    engine: str = "hf"
    batch_size: int = 16
    load_in_4bit: bool = False
    sampling: dict | None = None

    def __post_init__(self):
        if self.mode not in MODES:
            raise ValueError(f"mode must be one of {list(MODES)}")
        self.cfg = MODES[self.mode]
        self.thinking = self.cfg["thinking"]
        self.params = {**SAMPLING[self.thinking], **(self.sampling or {})}
        self._load()

    # ------------------------------------------------------------------
    def _load(self):
        if self.engine == "vllm":
            from vllm import LLM, SamplingParams  # type: ignore
            if self.adapters:
                raise ValueError("vllm 엔진은 병합된 모델 폴더를 model_path 로 주세요 (scripts/merge_adapters.py)")
            self.llm = LLM(model=self.model_path, dtype="bfloat16", seed=0)
            self._SP = SamplingParams
            from transformers import AutoTokenizer
            self.tok = AutoTokenizer.from_pretrained(self.model_path)
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(self.model_path)
        self.tok.padding_side = "left"
        if self.tok.pad_token is None:
            self.tok.pad_token = "<|endoftext|>"
        kw = {"dtype": torch.bfloat16 if torch.cuda.is_available() else torch.float32}
        if torch.cuda.is_available():
            kw["device_map"] = "auto"
        if self.load_in_4bit:
            from transformers import BitsAndBytesConfig
            kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                           bnb_4bit_compute_dtype=torch.bfloat16)
        model = AutoModelForCausalLM.from_pretrained(self.model_path, **kw)
        for i, a in enumerate(self.adapters):
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, a)
            if i < len(self.adapters) - 1 or not self.load_in_4bit:
                model = model.merge_and_unload()
        model.eval()
        self.model = model
        self.device = next(model.parameters()).device

    def set_mode(self, mode: str):
        """같은 가중치로 base / base_prompt / base_thinking 을 바꿔 가며 쓴다 (모델 다시 안 올림)."""
        if self.adapters and mode not in ("sft", "sft_dpo"):
            raise ValueError("adapter 가 붙은 모델은 base 모드로 바꿀 수 없다")
        self.mode, self.cfg = mode, MODES[mode]
        self.thinking = self.cfg["thinking"]
        self.params = {**SAMPLING[self.thinking], **(self.sampling or {})}

    # ------------------------------------------------------------------
    def prepare(self, messages: list[dict]) -> str:
        msgs = list(messages)
        if not self.cfg["system"]:
            msgs = [m for m in msgs if m["role"] != "system"]
        return render_prompt(msgs, thinking=self.thinking)

    def generate(self, batch: list[list[dict]], seed: int = 0) -> list[GenOutput]:
        prompts = [self.prepare(m) for m in batch]
        outs: list[GenOutput] = []
        for i in range(0, len(prompts), self.batch_size):
            outs.extend(self._gen(prompts[i:i + self.batch_size], seed + i))
        return outs

    def _gen(self, prompts: list[str], seed: int) -> list[GenOutput]:
        p = self.params
        if self.engine == "vllm":
            sp = self._SP(temperature=p["temperature"], top_p=p["top_p"], top_k=p["top_k"],
                          max_tokens=p["max_new_tokens"], seed=seed, stop=["<|im_end|>"])
            res = self.llm.generate(prompts, sp, use_tqdm=False)
            raws = [r.outputs[0].text for r in res]
        else:
            import torch
            torch.manual_seed(seed)
            enc = self.tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(self.device)
            sample_kw = ({"do_sample": True, "temperature": p["temperature"], "top_p": p["top_p"], "top_k": p["top_k"]}
                         if p["temperature"] > 0 else {"do_sample": False})
            with torch.no_grad():
                out = self.model.generate(
                    **enc, **sample_kw, max_new_tokens=p["max_new_tokens"],
                    pad_token_id=self.tok.pad_token_id,
                    eos_token_id=[self.tok.convert_tokens_to_ids("<|im_end|>"), self.tok.eos_token_id],
                )
            gen = out[:, enc["input_ids"].shape[1]:]
            raws = self.tok.batch_decode(gen, skip_special_tokens=False)
        results = []
        for raw in raws:
            raw = raw.split("<|im_end|>")[0].replace("<|endoftext|>", "")
            if self.thinking:
                th, ans = split_think(raw)
                if "</think>" not in raw:  # 생각이 max_new_tokens 안에 안 끝남
                    th, ans = raw, ""
            else:
                th, ans = "", split_think(raw)[1]
            results.append(GenOutput(text=ans.strip(), thinking=th, raw=raw))
        return results


def load_teacher(name: str, model_path: str, sft_adapter: str | None = None, dpo_adapter: str | None = None,
                 **kw) -> Teacher:
    """평가 대상 이름 -> Teacher. sft/sft_dpo 는 어댑터 경로가 필요하다."""
    adapters: list[str] = []
    if name in ("sft", "sft_dpo"):
        if not sft_adapter:
            raise ValueError(f"{name} needs --sft_adapter")
        adapters.append(sft_adapter)
    if name == "sft_dpo":
        if not dpo_adapter:
            raise ValueError("sft_dpo needs --dpo_adapter")
        adapters.append(dpo_adapter)
    return Teacher(model_path=model_path, mode=name, adapters=adapters, **kw)
