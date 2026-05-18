"""
Evaluation script — run on Kaggle after training is complete.
Compares base Mistral-7B vs your fine-tuned model on API code generation prompts.
Results go into evaluation_results.json which you include in your GitHub README.
"""

import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from rouge_score import rouge_scorer

BASE_MODEL_ID = "mistralai/Mistral-7B-v0.1"
FINETUNED_MODEL_ID = "your-hf-username/mistral-api-codegen"  # <-- change this

# Hand-crafted evaluation prompts covering different API patterns
EVAL_PROMPTS = [
    {
        "instruction": "Create a FastAPI POST endpoint at /users that accepts name and email fields, validates they are non-empty, and returns a JSON response with a success message and a generated user ID.",
        "category": "FastAPI - POST endpoint",
    },
    {
        "instruction": "Write an Express.js GET endpoint that fetches all products from a MongoDB collection and returns them as JSON. Include error handling for database connection failures.",
        "category": "Express.js - GET with MongoDB",
    },
    {
        "instruction": "Create a FastAPI middleware that checks for a Bearer token in the Authorization header and returns 401 if it is missing or invalid.",
        "category": "FastAPI - Auth middleware",
    },
    {
        "instruction": "Write a Flask REST API endpoint that accepts a CSV file upload, parses it using pandas, and returns summary statistics as JSON.",
        "category": "Flask - File upload",
    },
    {
        "instruction": "Create an Express.js route for user login that accepts email and password, verifies the password using bcrypt, and returns a signed JWT token on success.",
        "category": "Express.js - JWT auth",
    },
    {
        "instruction": "Write a FastAPI CRUD router for a 'Task' model with fields title, description, and status. Include endpoints for create, read all, read by id, update, and delete.",
        "category": "FastAPI - Full CRUD",
    },
    {
        "instruction": "Create a Node.js Express middleware that rate-limits requests to 100 per hour per IP address and returns 429 Too Many Requests when exceeded.",
        "category": "Express.js - Rate limiting",
    },
    {
        "instruction": "Write a Django REST Framework viewset for a Blog model with title, content, author, and created_at fields. Include pagination and filtering by author.",
        "category": "Django - ViewSet",
    },
]


def load_model(model_id, adapter_id=None):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token

    if adapter_id:
        model = PeftModel.from_pretrained(model, adapter_id)

    return model, tokenizer


def generate(model, tokenizer, instruction, max_new_tokens=300):
    prompt = f"### Instruction:\n{instruction}\n\n### Response:\n"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to("cuda")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.2,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    full = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return full.split("### Response:\n")[-1].strip()


def score_outputs(base_outputs, ft_outputs):
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    results = []
    for i, (base, ft, prompt) in enumerate(zip(base_outputs, ft_outputs, EVAL_PROMPTS)):
        # Use fine-tuned output as pseudo-reference (measures structural similarity)
        score = scorer.score(ft, base)
        results.append({
            "category": prompt["category"],
            "instruction": prompt["instruction"],
            "base_model_output": base,
            "finetuned_output": ft,
            "rougeL_ft_vs_base": round(score["rougeL"].fmeasure, 4),
        })
    return results


def main():
    print("Loading base model...")
    base_model, base_tokenizer = load_model(BASE_MODEL_ID)

    print("Generating base model outputs...")
    base_outputs = []
    for p in EVAL_PROMPTS:
        out = generate(base_model, base_tokenizer, p["instruction"])
        base_outputs.append(out)
        print(f"  [{p['category']}] done")

    del base_model
    torch.cuda.empty_cache()

    print("\nLoading fine-tuned model...")
    ft_model, ft_tokenizer = load_model(BASE_MODEL_ID, adapter_id=FINETUNED_MODEL_ID)

    print("Generating fine-tuned model outputs...")
    ft_outputs = []
    for p in EVAL_PROMPTS:
        out = generate(ft_model, ft_tokenizer, p["instruction"])
        ft_outputs.append(out)
        print(f"  [{p['category']}] done")

    results = score_outputs(base_outputs, ft_outputs)

    with open("evaluation_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n── EVALUATION SUMMARY ──────────────────────────────")
    for r in results:
        print(f"{r['category']:<40} ROUGE-L: {r['rougeL_ft_vs_base']}")
    print("\nFull results saved to evaluation_results.json")
    print("Include this file in your GitHub repo and screenshot outputs for README.")


if __name__ == "__main__":
    main()
