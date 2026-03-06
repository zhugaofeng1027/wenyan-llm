import argparse
import inspect
from pathlib import Path
from typing import List, Optional

import torch
from datasets import Dataset, DatasetDict, load_dataset
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from trl import SFTTrainer


def str2bool(v: str) -> bool:
    if isinstance(v, bool):
        return v
    val = v.lower().strip()
    if val in {"1", "true", "t", "yes", "y"}:
        return True
    if val in {"0", "false", "f", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {v}")


def pick_train_split(ds: DatasetDict, preferred: str) -> Dataset:
    if preferred in ds:
        return ds[preferred]
    if "train" in ds:
        return ds["train"]
    first_key = next(iter(ds.keys()))
    print(f"[WARN] Split '{preferred}' not found, using '{first_key}'.")
    return ds[first_key]


def find_string_columns(dataset: Dataset) -> List[str]:
    cols: List[str] = []
    for name, feature in dataset.features.items():
        dtype = getattr(feature, "dtype", None)
        if dtype == "string":
            cols.append(name)
    return cols


def choose_text_column(dataset: Dataset, explicit: Optional[str]) -> str:
    if explicit:
        if explicit not in dataset.column_names:
            raise ValueError(f"text_column '{explicit}' not in {dataset.column_names}")
        return explicit

    preferred = ["text", "content", "output", "response", "answer", "completion", "target"]
    available = set(dataset.column_names)
    for col in preferred:
        if col in available:
            return col

    string_cols = find_string_columns(dataset)
    if len(string_cols) == 1:
        return string_cols[0]

    if string_cols:
        sample = dataset.select(range(min(128, len(dataset))))
        best_col = string_cols[0]
        best_len = -1.0
        for col in string_cols:
            total = 0
            for row in sample:
                val = row.get(col)
                if isinstance(val, str):
                    total += len(val)
            avg_len = total / max(len(sample), 1)
            if avg_len > best_len:
                best_len = avg_len
                best_col = col
        print(f"[INFO] Auto-selected text column: {best_col}")
        return best_col

    raise ValueError(f"Cannot infer text column from columns: {dataset.column_names}")


def build_model_and_tokenizer(args: argparse.Namespace):
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    compute_dtype = torch.bfloat16 if use_bf16 else torch.float16

    quant_config = None
    if args.use_4bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=quant_config,
        dtype=compute_dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.use_cache = False

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if args.use_4bit:
        model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=args.target_modules,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model, tokenizer, use_bf16


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SFT/QLoRA training for Classical Chinese model.")
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--dataset_name", type=str, default="rick22630773/classical_chinese")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--text_column", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="outputs/qwen2.5-1.5b-instruct-wenyan-lora")

    parser.add_argument("--max_seq_length", type=int, default=512)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--max_steps", type=int, default=1200)
    parser.add_argument("--warmup_steps", type=int, default=None)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--weight_decay", type=float, default=0.01)

    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=200)

    parser.add_argument("--use_4bit", type=str2bool, default=True)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument(
        "--target_modules",
        nargs="+",
        default=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    parser.add_argument("--gradient_checkpointing", type=str2bool, default=True)
    parser.add_argument("--merge_and_save", type=str2bool, default=False)

    return parser.parse_args()


def build_sft_trainer(
    model,
    tokenizer,
    training_args: TrainingArguments,
    train_ds: Dataset,
    text_column: str,
    max_seq_length: int,
) -> SFTTrainer:
    supported = set(inspect.signature(SFTTrainer.__init__).parameters.keys())

    kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": train_ds,
    }

    if "tokenizer" in supported:
        kwargs["tokenizer"] = tokenizer
    elif "processing_class" in supported:
        kwargs["processing_class"] = tokenizer

    if "dataset_text_field" in supported:
        kwargs["dataset_text_field"] = text_column
    elif "formatting_func" in supported:

        def formatting_func(example):
            value = example[text_column]
            if isinstance(value, list):
                return [str(x) for x in value]
            return str(value)

        kwargs["formatting_func"] = formatting_func

    if "max_seq_length" in supported:
        kwargs["max_seq_length"] = max_seq_length
    if "packing" in supported:
        kwargs["packing"] = True

    return SFTTrainer(**kwargs)


def main() -> None:
    args = parse_args()

    print(f"[INFO] Loading dataset: {args.dataset_name}")
    loaded = load_dataset(args.dataset_name)
    if isinstance(loaded, DatasetDict):
        train_ds = pick_train_split(loaded, args.split)
    else:
        train_ds = loaded

    text_column = choose_text_column(train_ds, args.text_column)
    print(f"[INFO] Using text column: {text_column}")

    model, tokenizer, use_bf16 = build_model_and_tokenizer(args)

    warmup_steps = args.warmup_steps
    if warmup_steps is None:
        warmup_steps = max(1, int(args.max_steps * args.warmup_ratio))
    print(f"[INFO] warmup_steps = {warmup_steps}")

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        warmup_steps=warmup_steps,
        weight_decay=args.weight_decay,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=3,
        lr_scheduler_type="cosine",
        optim="paged_adamw_8bit" if args.use_4bit else "adamw_torch",
        fp16=not use_bf16,
        bf16=use_bf16,
        gradient_checkpointing=args.gradient_checkpointing,
        report_to="none",
        remove_unused_columns=False,
    )

    trainer = build_sft_trainer(
        model=model,
        tokenizer=tokenizer,
        training_args=training_args,
        train_ds=train_ds,
        text_column=text_column,
        max_seq_length=args.max_seq_length,
    )

    trainer.train()
    trainer.model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"[INFO] LoRA adapter saved to: {args.output_dir}")

    if args.merge_and_save:
        merge_dir = f"{args.output_dir}-merged"
        base_model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
        merged = PeftModel.from_pretrained(base_model, args.output_dir)
        merged = merged.merge_and_unload()
        Path(merge_dir).mkdir(parents=True, exist_ok=True)
        merged.save_pretrained(merge_dir)
        tokenizer.save_pretrained(merge_dir)
        print(f"[INFO] Merged model saved to: {merge_dir}")


if __name__ == "__main__":
    main()
