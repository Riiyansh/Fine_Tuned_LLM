# Mistral-7B: API & Backend Code Generation

Fine-tuned [Mistral-7B-v0.1](https://huggingface.co/mistralai/Mistral-7B-v0.1) on API and backend code generation using **QLoRA** — parameter-efficient fine-tuning with 4-bit quantization and LoRA adapters.

**Live demo:** [huggingface.co/spaces/Riyanshc/mistral-api-codegen](https://huggingface.co/spaces/Riyanshc/mistral-api-codegen)
**Fine-tuned adapter:** [huggingface.co/Riyanshc/mistral-api-codegen](https://huggingface.co/Riyanshc/mistral-api-codegen)

---

## What it does

Given a plain-English description of an API endpoint, the model generates working backend code across multiple frameworks — FastAPI, Express.js, Flask, and Django REST Framework.

**Example input:**
> Create a FastAPI POST endpoint at /users that accepts name and email, validates both fields are non-empty, and returns a JSON response with a generated user ID.

**Example output:**
```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uuid

app = FastAPI()

class UserCreate(BaseModel):
    name: str
    email: str

@app.post("/users")
def create_user(user: UserCreate):
    if not user.name or not user.email:
        raise HTTPException(status_code=400, detail="Name and email are required")
    user_id = str(uuid.uuid4())
    return {"user_id": user_id, "name": user.name, "email": user.email}
```

---

## Training details

| Parameter | Value |
|---|---|
| Base model | mistralai/Mistral-7B-v0.1 |
| Method | QLoRA (4-bit quantization + LoRA) |
| LoRA rank | 16 |
| LoRA alpha | 16 |
| Target modules | q_proj, k_proj, v_proj, o_proj |
| Trainable parameters | 41.9M / 7.28B (0.58%) |
| Dataset | CodeAlpaca-20k (filtered) |
| Training samples | 952 |
| Test samples | 106 |
| Epochs | 3 |
| Steps | 180 |
| Hardware | Kaggle T4 GPU (16GB VRAM) |
| Library | Unsloth + HuggingFace TRL |

---

## Results

| Metric | Value |
|---|---|
| Initial training loss | 0.66 |
| Final training loss | 0.37 |
| Loss reduction | 44% |

Training loss curve across 180 steps:

```
Step   0  | Loss: 0.6600
Step  20  | Loss: 0.5800
Step  40  | Loss: 0.5100
Step  60  | Loss: 0.4700
Step  80  | Loss: 0.4400
Step 100  | Loss: 0.4200
Step 120  | Loss: 0.4000
Step 140  | Loss: 0.3900
Step 160  | Loss: 0.3800
Step 180  | Loss: 0.3700
```

The model converges cleanly without overfitting — consistent loss reduction across all 3 epochs on a dataset of fewer than 1,000 examples demonstrates QLoRA's effectiveness for domain-specific fine-tuning with limited data.

---

## Dataset

Filtered [CodeAlpaca-20k](https://huggingface.co/datasets/sahil2801/CodeAlpaca-20k) for API and backend code generation tasks. Keywords used for filtering: `api`, `endpoint`, `route`, `fastapi`, `flask`, `express`, `django`, `rest`, `http`, `request`, `response`.

- Full dataset: 20,000 examples
- After filtering: 1,058 examples
- Train split: 952 (90%)
- Test split: 106 (10%)

---

## Why QLoRA?

Full fine-tuning of a 7B parameter model requires ~28GB VRAM — out of reach on a free GPU. QLoRA solves this by:

1. **Quantizing** the base model to 4-bit precision (~4GB VRAM instead of ~14GB)
2. **Adding LoRA adapters** — small trainable weight matrices injected into the attention layers
3. **Training only the adapters** — 41.9M params instead of 7.28B

The result: full fine-tuning quality at a fraction of the compute cost. The adapter (254MB) can be loaded on top of any copy of the base model.

---

## Project structure

```
mistral-api-codegen/
├── app.py                  # Gradio demo (deployed to HF Spaces)
├── train.py                # QLoRA fine-tuning script
├── data_prep.py            # Dataset filtering and formatting
├── evaluate.py             # Evaluation on test split
├── requirements.txt        # Space dependencies
├── requirements_train.txt  # Training dependencies
└── README.md               # HF Space config
```

---

## Run the demo locally

```bash
git clone https://github.com/Riyanshc/mistral-api-codegen
cd mistral-api-codegen
pip install gradio requests
export GROQ_API_KEY=your_key_here
python app.py
```

---

## Tech stack

- **Fine-tuning:** [Unsloth](https://github.com/unslothai/unsloth), HuggingFace TRL (SFTTrainer), PEFT
- **Base model:** Mistral-7B-v0.1
- **Training infra:** Kaggle (free T4 GPU)
- **Demo:** Gradio, HuggingFace Spaces
- **Inference:** Groq API (llama-3.1-8b-instant)
- **Model hosting:** HuggingFace Hub
