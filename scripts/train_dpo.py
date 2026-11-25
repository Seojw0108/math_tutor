"""⑦ DPO (TRL DPOTrainer). SFT 모델에서 시작, β 0.1, lr 5e-6, dpo.jsonl 약 3,000~5,000쌍.

  python scripts/train_dpo.py --config configs/dpo.yaml --set sft_adapter=runs/20261201_sft/adapter

구조: base + SFT 어댑터를 병합(=SFT 모델) -> 새 LoRA 를 붙여 DPO.
      기준(reference) 모델 = 새 LoRA 를 끈 모델 = SFT 모델 (TRL 이 peft 모델이면 어댑터를 꺼서 ref 로 쓴다).
추론: tutor.generation.load_teacher("sft_dpo", sft_adapter=..., dpo_adapter=...) 가 같은 순서로 붙인다.

프롬프트 렌더링은 SFT 와 똑같이 tutor.chat.render_prompt (모든 선생님 턴에 빈 think 블록).
chosen/rejected 는 내용만 넣고 EOS(<|im_end|>)는 TRL 이 붙인다.
"""
import argparse
import random
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch  # noqa: E402
from datasets import Dataset  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback  # noqa: E402

from tutor.chat import dpo_example  # noqa: E402
from tutor.io_utils import read_jsonl  # noqa: E402
from tutor.runlog import RunLogger, count_trainable, load_config, warmup_kwargs  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--config", default="configs/dpo.yaml")
ap.add_argument("--set", nargs="*", default=[])
ap.add_argument("--max_steps", type=int, default=-1)
a = ap.parse_args()
cfg = load_config(a.config, a.set)
T, L = cfg["train"], cfg["lora"]
random.seed(cfg["seed"])
torch.manual_seed(cfg["seed"])

tok = AutoTokenizer.from_pretrained(cfg["model_path"])
if tok.pad_token is None:
    tok.pad_token = "<|endoftext|>"

# ---------------- 데이터 ----------------
pairs = read_jsonl(cfg["data_path"])
rows, dropped = [], 0
for p in pairs:
    ex = dpo_example(p["prompt"], p["chosen"], p["rejected"])
    n = len(tok(ex["prompt"] + max(ex["chosen"], ex["rejected"], key=len), add_special_tokens=False)["input_ids"]) + 1
    if n > cfg["max_length"]:  # TRL 은 앞쪽을 남기고 자르므로 답이 잘린다 -> 미리 뺀다
        dropped += 1
        continue
    rows.append({**ex, "rejected_type": p.get("rejected_type", "")})
random.Random(cfg["seed"]).shuffle(rows)
n_val = max(1, int(len(rows) * cfg["val_ratio"])) if len(rows) >= 50 else 0
val_ds = Dataset.from_list(rows[:n_val]) if n_val else None
train_ds = Dataset.from_list(rows[n_val:])
types = Counter(r["rejected_type"] for r in rows)

# ---------------- 모델: base + SFT 병합 ----------------
use_cuda = torch.cuda.is_available()
bf16 = bool(T.get("bf16")) and use_cuda and torch.cuda.is_bf16_supported()
kw = {"dtype": torch.bfloat16 if bf16 else torch.float32}
if use_cuda:
    kw["device_map"] = {"": 0}
model = AutoModelForCausalLM.from_pretrained(cfg["model_path"], **kw)
from peft import LoraConfig, PeftModel  # noqa: E402

model = PeftModel.from_pretrained(model, cfg["sft_adapter"]).merge_and_unload()
model.config.use_cache = False

from trl import DPOConfig, DPOTrainer  # noqa: E402

log = RunLogger(cfg["name"], config=cfg)
log.log(event="data", n_train=len(train_ds), n_val=n_val, dropped_too_long=dropped, rejected_types=dict(types))
print(f"pairs train {len(train_ds)} / val {n_val} / dropped(too long) {dropped} / types {dict(types)}")

args = DPOConfig(
    output_dir=str(log.dir / "ckpt"), beta=T["beta"], max_length=cfg["max_length"],
    num_train_epochs=T["epochs"], max_steps=a.max_steps,
    learning_rate=T["learning_rate"], lr_scheduler_type=T["lr_scheduler"], **warmup_kwargs(DPOConfig, T["warmup_ratio"]),
    per_device_train_batch_size=T["per_device_batch_size"], per_device_eval_batch_size=T["per_device_batch_size"],
    gradient_accumulation_steps=T["grad_accum"], optim=T["optimizer"] if use_cuda else "adamw_torch",
    bf16=bf16, gradient_checkpointing=T["gradient_checkpointing"] and use_cuda,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    logging_steps=T["logging_steps"], save_steps=T["save_steps"], save_total_limit=2,
    eval_strategy="steps" if val_ds is not None else "no", eval_steps=T["eval_steps"],
    report_to="none", seed=cfg["seed"],
)


class LogCB(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **k):
        if logs:
            log.log(step=state.global_step, **{kk: v for kk, v in logs.items() if isinstance(v, (int, float))})


trainer = DPOTrainer(
    model=model, args=args, train_dataset=train_ds, eval_dataset=val_ds, processing_class=tok,
    peft_config=LoraConfig(r=L["r"], lora_alpha=L["alpha"], lora_dropout=L["dropout"],
                           target_modules=L["target_modules"], task_type="CAUSAL_LM"),
    callbacks=[LogCB()],
)
params = count_trainable(trainer.model)
t0 = time.time()
result = trainer.train()
train_sec = time.time() - t0
trainer.model.save_pretrained(log.dir / "adapter")
tok.save_pretrained(log.dir / "adapter")

last = {}
for h in trainer.state.log_history:
    last.update({k: v for k, v in h.items() if k.startswith(("rewards/", "eval_rewards/", "logps/"))})
summary = log.finish(
    method="DPO + LoRA" + ("(bf16)" if bf16 else ""), beta=T["beta"], learning_rate=T["learning_rate"],
    batch=T["per_device_batch_size"] * T["grad_accum"], epochs=T["epochs"], optimizer=args.optim,
    lora_r=L["r"], lora_alpha=L["alpha"], **params, train_sec=round(train_sec, 1),
    train_loss=round(result.training_loss, 4), final_reward_stats=last, sft_adapter=cfg["sft_adapter"],
    n_pairs=len(train_ds), why=cfg.get("why"),
)
print(summary)
if last.get("rewards/margins", 1) <= 0.05:
    print("⚠ rewards/margins 가 거의 0: lr 을 5e-5 로 올려 다시 시도 권장 (configs/dpo.yaml why 참고)")
print("adapter ->", log.dir / "adapter")
