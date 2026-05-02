"""Fine-tune Qwen with LoRA on ETF ChatML JSONL — expect on the order of ~10 hours on an Apple M2 Pro with 32 GB (dataset size and flags affect runtime)."""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
)


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _default_dataset_path() -> Path:
    return _project_root() / "model_output" / "Synthesis" / "finetune_all" / "all_tickers_finetune_qa_chatml.jsonl"


def _default_output_dir() -> Path:
    return _project_root() / "model_output" / "Synthesis" / "finetuned" / "qwen2.5_7b_lora"


def _render_chat(messages: Sequence[Dict[str, str]], tokenizer: AutoTokenizer) -> str:
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(
                list(messages),
                tokenize=False,
                add_generation_prompt=False,
            )
        except Exception:
            pass

    lines: List[str] = []
    for m in messages:
        role = str(m.get("role", "user")).upper()
        content = str(m.get("content", "")).strip()
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


@dataclass
class EncodedExample:
    input_ids: List[int]
    attention_mask: List[int]
    labels: List[int]


class ChatFineTuneDataset(Dataset):
    def __init__(self, rows: Sequence[EncodedExample]) -> None:
        self.rows = list(rows)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, List[int]]:
        row = self.rows[idx]
        return {
            "input_ids": row.input_ids,
            "attention_mask": row.attention_mask,
            "labels": row.labels,
        }


class ChatDataCollator:
    def __init__(self, tokenizer: PreTrainedTokenizerBase) -> None:
        self.tokenizer = tokenizer

    def __call__(self, features: Sequence[Dict[str, List[int]]]) -> Dict[str, torch.Tensor]:
        inputs = [{"input_ids": f["input_ids"], "attention_mask": f["attention_mask"]} for f in features]
        batch = self.tokenizer.pad(
            inputs,
            padding=True,
            return_tensors="pt",
        )
        labels = [torch.tensor(f["labels"], dtype=torch.long) for f in features]
        batch["labels"] = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)
        return batch


def _load_chatml_rows(path: Path) -> List[Dict[str, object]]:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    rows: List[Dict[str, object]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "messages" not in obj:
                continue
            rows.append(obj)
    if not rows:
        raise ValueError(f"No usable chat rows found in: {path}")
    return rows


def _encode_rows(
    rows: Sequence[Dict[str, object]],
    tokenizer: AutoTokenizer,
    max_length: int,
) -> List[EncodedExample]:
    encoded: List[EncodedExample] = []
    for row in rows:
        messages = row.get("messages", [])
        if not isinstance(messages, list) or not messages:
            continue
        text = _render_chat(messages, tokenizer=tokenizer)
        toks = tokenizer(
            text,
            truncation=True,
            max_length=max_length,
            padding=False,
        )
        input_ids = toks["input_ids"]
        attention_mask = toks["attention_mask"]
        labels = [tok if mask == 1 else -100 for tok, mask in zip(input_ids, attention_mask)]
        encoded.append(
            EncodedExample(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
        )
    if not encoded:
        raise ValueError("Tokenization produced zero training examples.")
    return encoded


def _train_val_split(
    data: Sequence[EncodedExample],
    eval_ratio: float,
    seed: int,
) -> tuple[List[EncodedExample], List[EncodedExample]]:
    n = len(data)
    if n < 2 or eval_ratio <= 0:
        return list(data), []
    idx = list(range(n))
    rng = random.Random(seed)
    rng.shuffle(idx)
    eval_n = max(1, int(n * eval_ratio))
    eval_idx = set(idx[:eval_n])
    train = [data[i] for i in range(n) if i not in eval_idx]
    eval_ = [data[i] for i in range(n) if i in eval_idx]
    return train, eval_


def _load_model_tokenizer(
    model_name: str,
    use_lora: bool,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    prefer_fp16: bool = False,
):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = {"low_cpu_mem_usage": True}
    if prefer_fp16:
        model_kwargs["torch_dtype"] = torch.float16
    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

    if use_lora:
        try:
            from peft import LoraConfig, TaskType, get_peft_model
        except Exception as exc:
            raise ImportError("LoRA requested but `peft` is not installed. Install with: pip install peft") from exc

        target_modules = [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
        peft_cfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules,
            bias="none",
        )
        model = get_peft_model(model, peft_cfg)
        model.print_trainable_parameters()

    return model, tokenizer


def run_finetune(
    dataset_path: Path,
    output_dir: Path,
    model_name: str = "Qwen/Qwen2.5-7B-Instruct",
    use_lora: bool = True,
    max_length: int = 1024,
    eval_ratio: float = 0.05,
    seed: int = 42,
    num_train_epochs: float = 1.0,
    learning_rate: float = 2e-5,
    per_device_train_batch_size: int = 1,
    gradient_accumulation_steps: int = 16,
    logging_steps: int = 10,
    save_steps: int = 200,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    max_samples: int = 0,
    gradient_checkpointing: bool = True,
) -> Dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = _load_chatml_rows(dataset_path)
    if max_samples and max_samples > 0:
        rows = rows[:max_samples]
    has_cuda = torch.cuda.is_available()
    has_mps = torch.backends.mps.is_available() if hasattr(torch.backends, "mps") else False

    model, tokenizer = _load_model_tokenizer(
        model_name=model_name,
        use_lora=use_lora,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        prefer_fp16=has_mps and not has_cuda,
    )
    if gradient_checkpointing:
        model.gradient_checkpointing_enable()
        # Reduces activation memory for checkpointed training.
        if hasattr(model, "config"):
            model.config.use_cache = False

    encoded = _encode_rows(rows=rows, tokenizer=tokenizer, max_length=max_length)
    train_rows, eval_rows = _train_val_split(encoded, eval_ratio=eval_ratio, seed=seed)

    train_ds = ChatFineTuneDataset(train_rows)
    eval_ds = ChatFineTuneDataset(eval_rows) if eval_rows else None

    train_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        logging_steps=logging_steps,
        save_steps=save_steps,
        eval_steps=save_steps if eval_ds is not None else None,
        eval_strategy="steps" if eval_ds is not None else "no",
        save_strategy="steps",
        bf16=has_cuda,
        fp16=False,
        report_to=[],
        dataloader_num_workers=0,
        dataloader_pin_memory=has_cuda,
        gradient_checkpointing=gradient_checkpointing,
        remove_unused_columns=False,
        seed=seed,
    )

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=ChatDataCollator(tokenizer=tokenizer),
    )
    trainer.train()

    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    summary = {
        "dataset_path": str(dataset_path),
        "output_dir": str(output_dir),
        "model_name": model_name,
        "use_lora": use_lora,
        "num_rows_raw": len(rows),
        "num_rows_encoded": len(encoded),
        "num_train_rows": len(train_rows),
        "num_eval_rows": len(eval_rows),
        "max_length": max_length,
        "max_samples": max_samples,
        "device": "cuda" if has_cuda else ("mps" if has_mps else "cpu"),
    }
    summary_path = output_dir / "finetune_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary["summary_json"] = str(summary_path)
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune Qwen on generated ETF ChatML dataset.")
    parser.add_argument("--dataset-path", type=Path, default=_default_dataset_path())
    parser.add_argument("--output-dir", type=Path, default=_default_output_dir())
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--eval-ratio", type=float, default=0.05)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=16)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--no-lora",
        action="store_true",
        help="Disable LoRA and full fine-tune all weights.",
    )
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--no-gradient-checkpointing",
        action="store_true",
        help="Disable gradient checkpointing (uses more memory, may be slightly faster).",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Optional cap on training rows for quick test",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    result = run_finetune(
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
        model_name=args.model_name,
        use_lora=not args.no_lora,
        max_length=args.max_length,
        eval_ratio=args.eval_ratio,
        seed=args.seed,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        max_samples=args.max_samples,
        gradient_checkpointing=not args.no_gradient_checkpointing,
    )
    print(json.dumps(result, indent=2))
