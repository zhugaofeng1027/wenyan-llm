import argparse
from pathlib import Path
from typing import Optional

import gradio as gr
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
    return (Path(path) / "adapter_config.json").exists()


def build_prompt(tokenizer, messages):
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    lines = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        if role == "system":
            lines.append(f"[System]\n{content}")
        elif role == "assistant":
            lines.append(f"[Assistant]\n{content}")
        else:
            lines.append(f"[User]\n{content}")
    lines.append("[Assistant]\n")
    return "\n\n".join(lines)


def load_model_and_tokenizer(base_model: str, checkpoint_dir: str, output_dir: str, use_4bit: bool):
    ckpt = checkpoint_dir.strip() if checkpoint_dir else ""
    if not ckpt:
        ckpt = find_latest_checkpoint(output_dir) or output_dir

    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    compute_dtype = torch.bfloat16 if use_bf16 else torch.float16

    quant_config = None
    if use_4bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
        )

    tokenizer_source = ckpt if (Path(ckpt) / "tokenizer_config.json").exists() else base_model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, trust_remote_code=True, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if is_lora_adapter(ckpt):
        base = AutoModelForCausalLM.from_pretrained(
            base_model,
            quantization_config=quant_config,
            dtype=compute_dtype,
            device_map="auto",
            trust_remote_code=True,
        )
        model = PeftModel.from_pretrained(base, ckpt)
        load_mode = "LoRA adapter"
    else:
        model = AutoModelForCausalLM.from_pretrained(
            ckpt,
            quantization_config=quant_config,
            dtype=compute_dtype,
            device_map="auto",
            trust_remote_code=True,
        )
        load_mode = "Merged/Full model"

    model.eval()
    return model, tokenizer, ckpt, load_mode


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gradio chatbot for base model + checkpoint.")
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--checkpoint_dir", type=str, default="")
    parser.add_argument("--output_dir", type=str, default="outputs/qwen2.5-1.5b-instruct-wenyan-lora")
    parser.add_argument("--use_4bit", type=str2bool, default=True)

    parser.add_argument("--system_prompt", type=str, default="你是一位精通古文与诗词创作的助手。")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--repetition_penalty", type=float, default=1.1)

    parser.add_argument("--server_name", type=str, default="127.0.0.1")
    parser.add_argument("--server_port", type=int, default=7860)
    parser.add_argument("--share", type=str2bool, default=False)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    with gr.Blocks(title="Wenyan LLM Chat") as demo:
        gr.Markdown("# Wenyan LLM Chat")

        model_state = gr.State({"model": None, "tokenizer": None, "messages": []})

        with gr.Row():
            base_model_in = gr.Textbox(label="Base Model", value=args.base_model)
            checkpoint_in = gr.Textbox(label="Checkpoint Dir (optional)", value=args.checkpoint_dir)
        with gr.Row():
            output_dir_in = gr.Textbox(label="Output Dir", value=args.output_dir)
            use_4bit_in = gr.Checkbox(label="Use 4-bit", value=args.use_4bit)

        with gr.Row():
            system_prompt_in = gr.Textbox(label="System Prompt", value=args.system_prompt)

        with gr.Row():
            max_new_tokens_in = gr.Slider(16, 1024, value=args.max_new_tokens, step=1, label="max_new_tokens")
            temperature_in = gr.Slider(0.1, 1.5, value=args.temperature, step=0.05, label="temperature")
            top_p_in = gr.Slider(0.1, 1.0, value=args.top_p, step=0.05, label="top_p")
            repetition_penalty_in = gr.Slider(1.0, 2.0, value=args.repetition_penalty, step=0.05, label="repetition_penalty")

        load_btn = gr.Button("加载模型", variant="primary")
        clear_btn = gr.Button("清空历史")
        status = gr.Markdown("状态: 未加载")

        chatbot = gr.Chatbot(label="对话", height=520)
        user_input = gr.Textbox(label="输入", placeholder="输入内容后回车", lines=2)
        send_btn = gr.Button("发送")

        def load_model_ui(base_model, checkpoint_dir, output_dir, use_4bit, system_prompt):
            model, tokenizer, used_ckpt, load_mode = load_model_and_tokenizer(
                base_model=base_model,
                checkpoint_dir=checkpoint_dir,
                output_dir=output_dir,
                use_4bit=use_4bit,
            )
            messages = []
            sys_prompt = (system_prompt or "").strip()
            if sys_prompt:
                messages.append({"role": "system", "content": sys_prompt})
            state = {"model": model, "tokenizer": tokenizer, "messages": messages}
            text = f"状态: 已加载 {load_mode} | 路径: `{used_ckpt}`"
            return state, [], text

        def clear_history_ui(state, system_prompt):
            messages = []
            sys_prompt = (system_prompt or "").strip()
            if sys_prompt:
                messages.append({"role": "system", "content": sys_prompt})
            state["messages"] = messages
            return state, []

        def chat_ui(message, history, state, base_model, checkpoint_dir, output_dir, use_4bit,
                    system_prompt, max_new_tokens, temperature, top_p, repetition_penalty):
            if not message or not message.strip():
                return "", history, state
            if history is None:
                history = []

            if state.get("model") is None or state.get("tokenizer") is None:
                model, tokenizer, _, _ = load_model_and_tokenizer(
                    base_model=base_model,
                    checkpoint_dir=checkpoint_dir,
                    output_dir=output_dir,
                    use_4bit=use_4bit,
                )
                state["model"] = model
                state["tokenizer"] = tokenizer
                if not state.get("messages"):
                    sys_prompt = (system_prompt or "").strip()
                    state["messages"] = [{"role": "system", "content": sys_prompt}] if sys_prompt else []

            model = state["model"]
            tokenizer = state["tokenizer"]
            messages = state.get("messages", [])

            messages.append({"role": "user", "content": message})
            prompt_text = build_prompt(tokenizer, messages)
            inputs = tokenizer(prompt_text, return_tensors="pt")
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=int(max_new_tokens),
                    do_sample=True,
                    temperature=float(temperature),
                    top_p=float(top_p),
                    repetition_penalty=float(repetition_penalty),
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id,
                )

            input_len = inputs["input_ids"].shape[-1]
            new_tokens = outputs[0][input_len:]
            answer = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            if not answer:
                answer = "<EMPTY>"

            messages.append({"role": "assistant", "content": answer})
            state["messages"] = messages

            history = history + [
                {"role": "user", "content": message},
                {"role": "assistant", "content": answer},
            ]
            return "", history, state

        load_btn.click(
            load_model_ui,
            inputs=[base_model_in, checkpoint_in, output_dir_in, use_4bit_in, system_prompt_in],
            outputs=[model_state, chatbot, status],
        )

        clear_btn.click(
            clear_history_ui,
            inputs=[model_state, system_prompt_in],
            outputs=[model_state, chatbot],
        )

        send_inputs = [
            user_input,
            chatbot,
            model_state,
            base_model_in,
            checkpoint_in,
            output_dir_in,
            use_4bit_in,
            system_prompt_in,
            max_new_tokens_in,
            temperature_in,
            top_p_in,
            repetition_penalty_in,
        ]
        send_outputs = [user_input, chatbot, model_state]

        send_btn.click(chat_ui, inputs=send_inputs, outputs=send_outputs)
        user_input.submit(chat_ui, inputs=send_inputs, outputs=send_outputs)

    demo.launch(server_name=args.server_name, server_port=args.server_port, share=args.share)


if __name__ == "__main__":
    main()
