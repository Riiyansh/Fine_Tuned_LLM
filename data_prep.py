"""
Dataset preparation for Mistral-7B API/backend code generation fine-tuning.
Run this locally to inspect and preview the dataset before uploading to Kaggle.
"""

from datasets import load_dataset
import json

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


def prepare_dataset():
    print("Loading CodeAlpaca-20k...")
    dataset = load_dataset("sahil2801/CodeAlpaca-20k", split="train")
    print(f"Total samples: {len(dataset)}")

    print("Filtering for API/backend related samples...")
    api_dataset = dataset.filter(is_api_related)
    print(f"API-related samples: {len(api_dataset)}")

    formatted = api_dataset.map(lambda x: {"text": format_instruction(x)})

    # Train/test split — 90/10
    split = formatted.train_test_split(test_size=0.1, seed=42)
    print(f"Train: {len(split['train'])} | Test: {len(split['test'])}")

    # Preview 3 samples
    print("\n--- SAMPLE PREVIEW ---")
    for i in range(min(3, len(split["train"]))):
        print(f"\n[Sample {i+1}]")
        print(split["train"][i]["text"][:500])
        print("...")

    # Save test set locally for evaluation later
    test_samples = [{"text": ex["text"]} for ex in split["test"]]
    with open("test_samples.json", "w") as f:
        json.dump(test_samples, f, indent=2)
    print(f"\nSaved {len(test_samples)} test samples to test_samples.json")

    return split


if __name__ == "__main__":
    prepare_dataset()
