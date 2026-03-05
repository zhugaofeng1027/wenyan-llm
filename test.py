import argparse
from pathlib import Path
from typing import Optional

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def str2bool(v: str) -> bool:
    if isinstance(v, bool):
        return v
    val = v.lower().strip()
    if val in {"1", "true", "t", "yes", "y"}:
        return True
    if val in {"0", "false", "f", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {v}")


def find_latest_checkpoint(output_dir: str) -> Optional[str]:
    root = Path(output_dir)
    if not root.exists():
        return None

    candidates = []
    for p in root.glob("checkpoint-*"):
        if p.is_dir():
            try:
                step = int(p.name.split("-")[-1])
                candidates.append((step, p))
            except ValueError:
                continue

    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0], reverse=True)
    return str(candidates[0][1])


def is_lora_adapter(path: str) -> bool:
    p = Path(path)
    return (p / "adapter_config.json").exists()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test Wenyan model checkpoint/adapter by generation.")
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-1.5B")
    parser.add_argument("--checkpoint_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="outputs/qwen2.5-1.5b-wenyan-lora")
    parser.add_argument("--use_4bit", type=str2bool, default=True)

    parser.add_argument("--prompt", type=str, default="请以文言文作五言绝句，题为《春夜》。")
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--repetition_penalty", type=float, default=1.1)

    return parser.parse_args()


def load_model_and_tokenizer(args: argparse.Namespace):
    ckpt = args.checkpoint_dir
    if ckpt is None:
        ckpt = find_latest_checkpoint(args.output_dir)
        if ckpt is None:
            ckpt = args.output_dir
    print(f"[INFO] Using checkpoint/model path: {ckpt}")

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

    tokenizer_source = ckpt if Path(ckpt).joinpath("tokenizer_config.json").exists() else args.base_model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, trust_remote_code=True, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if is_lora_adapter(ckpt):
        print("[INFO] Detected LoRA adapter checkpoint, loading base + adapter...")
        base_model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            quantization_config=quant_config,
            torch_dtype=compute_dtype,
            device_map="auto",
            trust_remote_code=True,
        )
        model = PeftModel.from_pretrained(base_model, ckpt)
    else:
        print("[INFO] Detected full/merged model checkpoint, loading directly...")
        model = AutoModelForCausalLM.from_pretrained(
            ckpt,
            quantization_config=quant_config,
            torch_dtype=compute_dtype,
            device_map="auto",
            trust_remote_code=True,
        )

    model.eval()
    return model, tokenizer


def main() -> None:
    args = parse_args()
    model, tokenizer = load_model_and_tokenizer(args)

    inputs = tokenizer(args.prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=True,
            temperature=args.temperature,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )

    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("\n===== Prompt =====")
    print(args.prompt)
    print("\n===== Output =====")
    print(text)


if __name__ == "__main__":
    main()
