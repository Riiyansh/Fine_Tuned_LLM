"""
Mistral-7B QLoRA Fine-tuning for API/Backend Code Generation
Run this on Kaggle (T4 GPU). Paste each section into a separate notebook cell.

Before running:
1. Enable GPU in Kaggle: Settings > Accelerator > GPU T4 x2
2. Add HuggingFace token secret in Kaggle: Settings > Secrets > HF_TOKEN
3. Add your HuggingFace username below
"""

HF_USERNAME = "your-hf-username"   # <-- change this
MODEL_ID = "mistralai/Mistral-7B-v0.1"
OUTPUT_MODEL_NAME = f"{HF_USERNAME}/mistral-api-codegen"

# ── CELL 1: Install dependencies ──────────────────────────────────────────────
"""
!pip install -q transformers==4.40.0 peft==0.10.0 trl==0.8.6 \
    bitsandbytes==0.43.1 accelerate==0.29.3 datasets==2.19.0 \
    huggingface_hub
"""

# ── CELL 2: Imports ───────────────────────────────────────────────────────────
import os
import json
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, TaskType, get_peft_model
from trl import SFTTrainer, SFTConfig
from huggingface_hub import login

# Login to HuggingFace using Kaggle secret
from kaggle_secrets import UserSecretsClient
secrets = UserSecretsClient()
hf_token = secrets.get_secret("HF_TOKEN")
login(token=hf_token)

print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")


# ── CELL 3: Dataset preparation ───────────────────────────────────────────────
API_KEYWORDS = [
    "api", "rest", "fastapi", "flask", "express", "endpoint", "route",
    "backend", "server", "http", "request", "response", "node.js", "nodejs",
    "middleware", "controller", "handler", "webhook", "jwt", "auth",
    "crud", "get request", "post request", "put request", "delete request",
    "json response", "status code", "rest api", "web server", "django",
]

def is_api_related(example):
    text = (example["instruction"] + " " + example.get("input", "")).lower()
    return any(kw in text for kw in API_KEYWORDS)

def format_instruction(example):
    if example.get("input", "").strip():
        return (
            f"### Instruction:\n{example['instruction']}\n\n"
            f"### Input:\n{example['input']}\n\n"
            f"### Response:\n{example['output']}"
        )
    return (
        f"### Instruction:\n{example['instruction']}\n\n"
        f"### Response:\n{example['output']}"
    )

print("Loading and filtering dataset...")
raw_dataset = load_dataset("sahil2801/CodeAlpaca-20k", split="train")
api_dataset = raw_dataset.filter(is_api_related)
formatted = api_dataset.map(lambda x: {"text": format_instruction(x)})
split = formatted.train_test_split(test_size=0.1, seed=42)

print(f"Train samples: {len(split['train'])}")
print(f"Test samples:  {len(split['test'])}")
print("\nSample:")
print(split["train"][0]["text"][:400])


# ── CELL 4: Load model with QLoRA ─────────────────────────────────────────────
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

print(f"Loading {MODEL_ID}...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)
model.config.use_cache = False
model.config.pretraining_tp = 1

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

print("Model loaded successfully.")


# ── CELL 5: LoRA config ───────────────────────────────────────────────────────
lora_config = LoraConfig(
    r=16,                    # rank — higher = more params, more capacity
    lora_alpha=32,           # scaling factor (typically 2x rank)
    target_modules=[         # which layers to adapt
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# Expected output: ~0.7% of params trainable — that's the point of LoRA


# ── CELL 6: Training ──────────────────────────────────────────────────────────
training_args = SFTConfig(
    output_dir="./mistral-api-codegen-checkpoints",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,       # effective batch = 4 * 4 = 16
    gradient_checkpointing=True,         # saves VRAM at cost of speed
    optim="paged_adamw_32bit",
    learning_rate=2e-4,
    weight_decay=0.001,
    fp16=True,
    bf16=False,
    max_grad_norm=0.3,
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
    logging_steps=10,
    save_steps=50,
    save_total_limit=2,
    max_seq_length=512,
    dataset_text_field="text",
    report_to="none",                    # set to "wandb" if you want W&B tracking
)

trainer = SFTTrainer(
    model=model,
    train_dataset=split["train"],
    eval_dataset=split["test"],
    peft_config=lora_config,
    tokenizer=tokenizer,
    args=training_args,
)

print("Starting training...")
trainer.train()
print("Training complete.")


# ── CELL 7: Save and push to HuggingFace Hub ──────────────────────────────────
trainer.model.save_pretrained("mistral-api-codegen-final")
tokenizer.save_pretrained("mistral-api-codegen-final")

trainer.model.push_to_hub(OUTPUT_MODEL_NAME, token=hf_token)
tokenizer.push_to_hub(OUTPUT_MODEL_NAME, token=hf_token)

print(f"Model pushed to: https://huggingface.co/{OUTPUT_MODEL_NAME}")


# ── CELL 8: Quick inference test ──────────────────────────────────────────────
from peft import PeftModel
from transformers import pipeline

def generate_code(instruction, max_new_tokens=256):
    prompt = f"### Instruction:\n{instruction}\n\n### Response:\n"
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.2,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response.split("### Response:\n")[-1].strip()

test_prompt = "Create a FastAPI endpoint that accepts a POST request with name and email, validates the input, and returns a success message."
print("Test prompt:", test_prompt)
print("\nGenerated code:")
print(generate_code(test_prompt))
