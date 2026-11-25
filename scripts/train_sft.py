"""⑥ SFT (LoRA). assistant(선생님) 토큰에만 손실 (tutor.chat.tokenize_for_sft).

  # 샘플 50개로 코드 점검: 손실이 내려가고 출력이 선생님 말투가 되는지만
  python scripts/train_sft.py --config configs/sft.yaml --set name=sft_check data_path=data/sample50.jsonl data_mode=raw train.epochs=5
  # 본 학습
  python scripts/train_sft.py --config configs/sft.yaml
  # 추가 실험 (⑧)
  python scripts/train_sft.py --config configs/sft.yaml --set name=sft_raw data_mode=raw
  python scripts/train_sft.py --config configs/sft.yaml --set name=sft_frac25 fraction=0.25
  python scripts/train_sft.py --config configs/sft.yaml --set name=sft_r8 lora.r=8 lora.alpha=16
  python scripts/train_sft.py --config configs/sft.yaml --set name=sft_qlora qlora=true

결과: runs/{날짜}_{name}/ 에 config.yaml, log.jsonl, summary.json, adapter/, samples.jsonl
"""
import argparse
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch  # noqa: E402
from transformers import (AutoModelForCausalLM, AutoTokenizer, Trainer, TrainerCallback,  # noqa: E402
                          TrainingArguments)

from tutor.chat import IGNORE_INDEX, tokenize_for_sft  # noqa: E402
from tutor.io_utils import read_jsonl, select_sft_dialogues, write_jsonl  # noqa: E402
from tutor.runlog import RunLogger, count_trainable, load_config, warmup_kwargs  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--config", default="configs/sft.yaml")
ap.add_argument("--set", nargs="*", default=[], help="key=value 덮어쓰기 (lora.r=8)")
ap.add_argument("--max_steps", type=int, default=-1, help="점검용")
a = ap.parse_args()
cfg = load_config(a.config, a.set)
T, L = cfg["train"], cfg["lora"]
random.seed(cfg["seed"])
torch.manual_seed(cfg["seed"])

# ---------------- 데이터 ----------------
dialogues = read_jsonl(cfg["data_path"])
if all(d.get("split") is None for d in dialogues):  # 샘플 50개처럼 split 이 없으면 모두 train 으로
    for d in dialogues:
        d["split"] = "train"
rows = select_sft_dialogues(dialogues, cfg["data_mode"], cfg["fraction"], cfg["seed"])
if not rows:
    sys.exit("학습할 대화가 없습니다 (split=train / filters.passed 확인)")
n_val = max(1, int(len(rows) * cfg["val_ratio"])) if len(rows) >= 20 else 0
val_rows, train_rows = rows[:n_val], rows[n_val:]

tok = AutoTokenizer.from_pretrained(cfg["model_path"])
if tok.pad_token is None:
    tok.pad_token = "<|endoftext|>"
train_ds = [tokenize_for_sft(d["messages"], tok, cfg["max_length"]) for d in train_rows]
val_ds = [tokenize_for_sft(d["messages"], tok, cfg["max_length"]) for d in val_rows]
lens = [len(x["input_ids"]) for x in train_ds]
n_trunc = sum(1 for x in lens if x >= cfg["max_length"])
n_teacher_turns = sum(sum(1 for m in d["messages"] if m["role"] == "assistant") for d in train_rows)


def collate(batch):
    m = max(len(x["input_ids"]) for x in batch)
    pad = tok.pad_token_id
    return {
        "input_ids": torch.tensor([x["input_ids"] + [pad] * (m - len(x["input_ids"])) for x in batch]),
        "labels": torch.tensor([x["labels"] + [IGNORE_INDEX] * (m - len(x["labels"])) for x in batch]),
        "attention_mask": torch.tensor([x["attention_mask"] + [0] * (m - len(x["attention_mask"])) for x in batch]),
    }


# ---------------- 모델 ----------------
use_cuda = torch.cuda.is_available()
bf16 = bool(T.get("bf16")) and use_cuda and torch.cuda.is_bf16_supported()
kw = {"dtype": torch.bfloat16 if bf16 else torch.float32}
if cfg.get("qlora"):
    from transformers import BitsAndBytesConfig
    kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                   bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)
if use_cuda:
    kw["device_map"] = {"": 0}
model = AutoModelForCausalLM.from_pretrained(cfg["model_path"], **kw)
model.config.use_cache = False

from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training  # noqa: E402

if cfg.get("qlora"):
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=T["gradient_checkpointing"])
model = get_peft_model(model, LoraConfig(r=L["r"], lora_alpha=L["alpha"], lora_dropout=L["dropout"],
                                         target_modules=L["target_modules"], task_type="CAUSAL_LM"))
params = count_trainable(model)

log = RunLogger(cfg["name"], config=cfg)
log.log(event="data", n_train=len(train_ds), n_val=len(val_ds), teacher_turns=n_teacher_turns,
        mean_tokens=round(sum(lens) / len(lens), 1), max_tokens=max(lens), truncated=n_trunc, **params)
print(f"train {len(train_ds)} dialogues ({n_teacher_turns} teacher turns), val {len(val_ds)}, "
      f"mean len {sum(lens) / len(lens):.0f}, truncated {n_trunc}, trainable {params}")


class LogCB(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **k):
        if logs:
            log.log(step=state.global_step, **{kk: v for kk, v in logs.items() if isinstance(v, (int, float))})


args = TrainingArguments(
    output_dir=str(log.dir / "ckpt"),
    num_train_epochs=T["epochs"], max_steps=a.max_steps,
    learning_rate=T["learning_rate"], lr_scheduler_type=T["lr_scheduler"], **warmup_kwargs(TrainingArguments, T["warmup_ratio"]),
    per_device_train_batch_size=T["per_device_batch_size"], per_device_eval_batch_size=T["per_device_batch_size"],
    gradient_accumulation_steps=T["grad_accum"], weight_decay=T.get("weight_decay", 0.0),
    optim=T["optimizer"] if use_cuda else "adamw_torch",
    bf16=bf16, gradient_checkpointing=T["gradient_checkpointing"] and use_cuda,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    logging_steps=T["logging_steps"], save_steps=T["save_steps"], save_total_limit=2,
    eval_strategy="steps" if val_ds else "no", eval_steps=T["eval_steps"],
    report_to="none", seed=cfg["seed"], remove_unused_columns=False, dataloader_num_workers=0,
)
trainer = Trainer(model=model, args=args, train_dataset=train_ds, eval_dataset=val_ds or None,
                  data_collator=collate, callbacks=[LogCB()])
t0 = time.time()
result = trainer.train()
train_sec = time.time() - t0
model.save_pretrained(log.dir / "adapter")
tok.save_pretrained(log.dir / "adapter")

# ---- 말투 확인용 샘플 몇 개 (검증 대화의 첫 선생님 턴 이후) ----
from tutor.chat import render_prompt  # noqa: E402

model.eval()
model.config.use_cache = True
samples = []
for d in (val_rows or train_rows)[:4]:
    msgs = d["messages"]
    cut = next((i for i in range(len(msgs)) if msgs[i].get("response_type") in ("오개념오답", "계산실수")), 3)
    prompt = render_prompt(msgs[:cut + 1])
    enc = tok(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=120, do_sample=False, pad_token_id=tok.pad_token_id,
                             eos_token_id=tok.convert_tokens_to_ids("<|im_end|>"))
    gen = tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)
    samples.append({"id": d["id"], "child": msgs[cut]["content"], "generated": gen.strip(),
                    "reference": msgs[cut + 1]["content"] if cut + 1 < len(msgs) else None})
write_jsonl(log.dir / "samples.jsonl", samples)

eval_metrics = trainer.evaluate() if val_ds else {}
summary = log.finish(
    method="QLoRA(4bit)" if cfg.get("qlora") else "LoRA(bf16)" if bf16 else "LoRA(fp32)",
    learning_rate=T["learning_rate"], batch=T["per_device_batch_size"] * T["grad_accum"],
    per_device_batch=T["per_device_batch_size"], grad_accum=T["grad_accum"], epochs=T["epochs"],
    optimizer=args.optim, lora_r=L["r"], lora_alpha=L["alpha"], **params,
    train_sec=round(train_sec, 1), train_loss=round(result.training_loss, 4),
    eval_loss=eval_metrics.get("eval_loss"), n_train=len(train_ds), data_mode=cfg["data_mode"],
    fraction=cfg["fraction"], why=cfg.get("why"),
)
print(summary)
print("adapter ->", log.dir / "adapter")
