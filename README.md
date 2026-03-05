# Wenyan LLM Quick Start (Qwen2.5-1.5B + Classical Chinese + QLoRA)

> Target GPU: RTX 4060 (8GB)

## 1) Create env and install

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
pip install -r requirements.txt
```

## 2) Start training

```powershell
python train_lora.py
```

Default setup:
- Base model: `Qwen/Qwen2.5-1.5B`
- Dataset: `rick22630773/classical_chinese`
- Method: `SFT + QLoRA (4-bit)`
- Output dir: `outputs/qwen2.5-1.5b-wenyan-lora`

## 3) Common overrides

```powershell
python train_lora.py --max_steps 1000 --learning_rate 1e-4 --max_seq_length 1024
python train_lora.py --dataset_name rick22630773/classical_chinese --split train
python train_lora.py --use_4bit false   # if you have enough VRAM
```

## 4) Merge LoRA (optional)

```powershell
python train_lora.py --merge_and_save true
```

Merged model will be saved to: `outputs/qwen2.5-1.5b-wenyan-lora-merged`
