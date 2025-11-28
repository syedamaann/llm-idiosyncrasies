import gradio as gr
import torch
import numpy as np
from transformers import AutoConfig, AutoModel, AutoTokenizer
from peft import PeftModel
from llm2vec import LLM2Vec
import gc
import os

# Global model variable
model = None

def load_classifier(checkpoint_path="./classifier_chat", num_labels=5):
    """Load the pre-trained LLM2Vec classifier."""
    print("Loading classifier...")

    torch.cuda.empty_cache()
    gc.collect()

    base_model_name = "McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp"

    config = AutoConfig.from_pretrained(
        base_model_name,
        trust_remote_code=True,
    )

    device_map = {"": 0} if torch.cuda.is_available() else "cpu"
    max_memory = {0: "14GiB", "cpu": "30GiB"} if torch.cuda.is_available() else None

    # Load base model
    model = AutoModel.from_pretrained(
        base_model_name,
        config=config,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
        max_memory=max_memory,
        offload_folder="./offload",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )

    # Load first adapter
    model = PeftModel.from_pretrained(
        model,
        base_model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    # Merge on CPU
    model = model.cpu()
    torch.cuda.empty_cache()
    gc.collect()

    model = model.merge_and_unload()
    model = model.to(torch.bfloat16)

    # Re-dispatch to GPU
    if torch.cuda.is_available():
        from accelerate import dispatch_model, infer_auto_device_map
        device_map_merged = infer_auto_device_map(
            model,
            max_memory=max_memory,
            no_split_module_classes=["LlamaDecoderLayer"]
        )
        model = dispatch_model(model, device_map=device_map_merged, offload_dir="./offload")

    torch.cuda.empty_cache()
    gc.collect()

    # Load supervised adapter
    model = PeftModel.from_pretrained(
        model,
        f"{base_model_name}-supervised",
        is_trainable=True,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    model = LLM2Vec(model, tokenizer, pooling_mode="mean", max_length=512)

    # Add classification head
    hidden_size = list(model.modules())[-1].weight.shape[0]
    model.head = torch.nn.Linear(hidden_size, num_labels, dtype=torch.bfloat16)

    head_path = os.path.join(checkpoint_path, "head.pt")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.head.load_state_dict(torch.load(head_path, map_location=device))
    model.head = model.head.to(device)

    model.eval()

    print("✓ Classifier loaded!")
    return model


def predict(text):
    """Predict which LLM generated the given text."""
    global model

    if model is None:
        return "Error: Model not loaded. Please wait for initialization.", None

    if not text.strip():
        return "Please enter some text to classify.", None

    label_names = ["ChatGPT", "Claude", "Grok", "Gemini", "DeepSeek"]

    # Prepare text
    prepared_text = model.prepare_for_tokenization(text)
    inputs = model.tokenize([prepared_text])

    # Move to device
    if torch.cuda.is_available():
        target_device = torch.device("cuda:0")
    else:
        target_device = next(model.parameters()).device

    inputs = {k: v.to(target_device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

    # Predict
    with torch.no_grad():
        embeddings = model.forward(inputs)

        if hasattr(model, 'head'):
            head_device = next(model.head.parameters()).device
            embeddings = embeddings.to(head_device)

        embeddings = embeddings.to(torch.bfloat16)
        logits = model.head(embeddings)
        probabilities = torch.nn.functional.softmax(logits, dim=-1)

    pred_label = torch.argmax(probabilities, dim=-1).item()
    all_probs = probabilities[0].float().cpu().numpy()

    # Create results
    result_text = f"**Most likely source: {label_names[pred_label]}**\n\n"
    result_text += f"**Confidence: {all_probs[pred_label]*100:.2f}%**\n\n"
    result_text += "### All Probabilities:\n\n"

    sorted_indices = np.argsort(all_probs)[::-1]
    for idx in sorted_indices:
        result_text += f"- **{label_names[idx]}**: {all_probs[idx]*100:.2f}%\n"

    # Create probability distribution for plot
    prob_dict = {label_names[i]: float(all_probs[i]) for i in range(len(label_names))}

    return result_text, prob_dict


# Examples
examples = [
    ["Hello! I'd be happy to help you with that question. Let me break this down into a few key points: First, it's important to understand the context. Second, we should consider the implications. Finally, let's look at practical applications."],
    ["Sure, I can help with that! Here's what you need to know..."],
    ["I'd be delighted to assist you with this matter. Allow me to provide a comprehensive explanation."],
]


# Create Gradio interface
with gr.Blocks(title="LLM Detector", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 🔍 Chat API LLM Detector

        Detect which AI model (ChatGPT, Claude, Grok, Gemini, or DeepSeek) generated a piece of text.

        **Accuracy: 97.1%** on test data | Based on [research paper](https://arxiv.org/abs/2502.12150)
        """
    )

    with gr.Row():
        with gr.Column(scale=2):
            text_input = gr.Textbox(
                label="Enter text to classify",
                placeholder="Paste the AI-generated text here...",
                lines=10,
            )

            submit_btn = gr.Button("🔍 Detect LLM", variant="primary", size="lg")

            gr.Markdown("### 📝 Examples")
            gr.Examples(
                examples=examples,
                inputs=text_input,
            )

        with gr.Column(scale=1):
            result_output = gr.Markdown(label="Results")
            plot_output = gr.BarPlot(
                x="Model",
                y="Probability",
                title="Prediction Confidence",
                y_lim=[0, 1],
            )

    submit_btn.click(
        fn=predict,
        inputs=text_input,
        outputs=[result_output, plot_output]
    )

    gr.Markdown(
        """
        ---
        ### ℹ️ How it works

        This classifier uses [LLM2Vec](https://arxiv.org/abs/2404.05961) (fine-tuned Llama 3 8B) to analyze writing patterns and predict the source model.

        **Limitations:**
        - Works best on unedited AI-generated text
        - Accuracy decreases for very short text (< 50 words) or heavily edited content

        **Citation:** [Idiosyncrasies in Large Language Models](https://arxiv.org/abs/2502.12150)
        """
    )


if __name__ == "__main__":
    # Load model on startup
    print("Initializing model... This may take a few minutes.")
    model = load_classifier()
    print("Ready!")

    # Launch app
    demo.launch(share=True)
