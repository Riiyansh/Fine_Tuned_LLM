import os
import gradio as gr
from huggingface_hub import InferenceClient

HF_TOKEN = os.environ.get("HFTOKEN")
client = InferenceClient(model="HuggingFaceH4/zephyr-7b-beta", token=HF_TOKEN)

EXAMPLES = [
    "Create a FastAPI POST endpoint at /users that accepts name and email, validates both fields are non-empty, and returns a JSON response with a generated user ID.",
    "Write an Express.js GET endpoint that fetches all products from MongoDB and returns them as JSON with error handling.",
    "Create a FastAPI middleware that validates a Bearer token in the Authorization header and returns 401 if missing or invalid.",
    "Write a Flask endpoint that accepts a JSON body with title and content, saves it to SQLite, and returns the created record.",
    "Create an Express.js route for user login with bcrypt password verification and JWT token generation on success.",
    "Build a Django REST Framework ViewSet for a Product model with CRUD operations and pagination.",
]


def generate(instruction, max_new_tokens=350):
    if not instruction.strip():
        return "Please enter an instruction."
    prompt = f"<|system|>\nYou are a backend API code generator. Write clean, working code only.</s>\n<|user|>\n{instruction}</s>\n<|assistant|>\n"
    try:
        response = client.text_generation(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=0.2,
            stop_sequences=["<|user|>", "</s>"],
        )
        return response.strip()
    except Exception as e:
        return f"Error: {str(e)}"


with gr.Blocks(title="Mistral-7B API CodeGen", theme=gr.themes.Monochrome()) as demo:
    gr.Markdown(
        """
        # Mistral-7B: API & Backend Code Generation
        Fine-tuned on 952 API/backend code generation examples using **QLoRA** (LoRA rank 16, 3 epochs).
        Training loss: **0.66 → 0.37** | Trainable params: **41.9M / 7.28B (0.58%)**

        Fine-tuned adapter: [riyansh-headout/mistral-api-codegen](https://huggingface.co/riyansh-headout/mistral-api-codegen)
        """
    )

    with gr.Row():
        instruction_box = gr.Textbox(
            label="Describe the API endpoint you want to build",
            placeholder="e.g. Create a FastAPI POST endpoint that accepts name and email...",
            lines=4,
        )

    generate_btn = gr.Button("Generate Code", variant="primary", size="lg")
    output = gr.Code(label="Generated Code", language="python", lines=25)

    gr.Examples(
        examples=EXAMPLES,
        inputs=instruction_box,
        label="Example prompts — click to load",
    )

    generate_btn.click(fn=generate, inputs=instruction_box, outputs=output)

if __name__ == "__main__":
    demo.launch()
