import argparse
import json
import os
import tempfile
import torch
import numpy as np
from transformers import AutoConfig, AutoModel, AutoTokenizer
from peft import PeftModel
from llm2vec import LLM2Vec


def load_classifier(checkpoint_path, num_labels=5):
    """Load the pre-trained LLM2Vec classifier."""
    print("Loading classifier...")

    # Load the base LLM2Vec model
    base_model_name = "McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp"

    config = AutoConfig.from_pretrained(
        base_model_name,
        trust_remote_code=True,
    )
    model = AutoModel.from_pretrained(
        base_model_name,
        config=config,
        torch_dtype=torch.bfloat16,
        device_map="cuda" if torch.cuda.is_available() else "cpu",
        trust_remote_code=True,
    )

    # Load PEFT adaptors
    model = PeftModel.from_pretrained(
        model,
        base_model_name,
        torch_dtype=torch.bfloat16,
        device_map="cuda" if torch.cuda.is_available() else "cpu",
        trust_remote_code=True,
    )
    model = model.merge_and_unload()

    model = PeftModel.from_pretrained(
        model,
        f"{base_model_name}-supervised",
        is_trainable=True,
        torch_dtype=torch.bfloat16,
        device_map="cuda" if torch.cuda.is_available() else "cpu",
        trust_remote_code=True,
    )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)

    # Create LLM2Vec model
    model = LLM2Vec(model, tokenizer, pooling_mode="mean", max_length=512)

    # Add classification head
    hidden_size = list(model.modules())[-1].weight.shape[0]
    model.head = torch.nn.Linear(hidden_size, num_labels, dtype=torch.bfloat16)

    # Load the trained classification head
    head_path = os.path.join(checkpoint_path, "head.pt")
    if os.path.exists(head_path):
        model.head.load_state_dict(torch.load(head_path, map_location="cuda" if torch.cuda.is_available() else "cpu"))
        print(f"Loaded classification head from {head_path}")
    else:
        raise FileNotFoundError(f"Classification head not found at {head_path}")

    model.eval()
    return model


def predict_text(model, text, label_names):
    """Predict which LLM generated the given text."""
    # Prepare text for the model
    prepared_text = model.prepare_for_tokenization(text)

    # Tokenize
    inputs = model.tokenize([prepared_text])

    # Move inputs to the same device as model
    device = next(model.parameters()).device
    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

    # Get embeddings and predict
    with torch.no_grad():
        embeddings = model.forward(inputs).to(torch.bfloat16)
        logits = model.head(embeddings)
        probabilities = torch.nn.functional.softmax(logits, dim=-1)

    # Get prediction
    pred_label = torch.argmax(probabilities, dim=-1).item()
    pred_prob = probabilities[0, pred_label].item()

    # Get all probabilities
    all_probs = probabilities[0].cpu().numpy()

    return pred_label, pred_prob, all_probs


def main():
    parser = argparse.ArgumentParser(description="Classify which LLM generated a piece of text (Chat APIs only)")
    parser.add_argument("--text", type=str, default=None, help="The text to classify (or will prompt for input)")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the classifier checkpoint directory")

    args = parser.parse_args()

    # Chat APIs only: ChatGPT, Claude, Grok, Gemini, DeepSeek
    label_names = ["ChatGPT", "Claude", "Grok", "Gemini", "DeepSeek"]
    num_labels = 5

    # Get text input
    if args.text is None:
        print("\nEnter the text to classify (press Enter twice when done):")
        lines = []
        while True:
            line = input()
            if line == "" and len(lines) > 0 and lines[-1] == "":
                break
            lines.append(line)
        text = "\n".join(lines[:-1])  # Remove the last empty line
    else:
        text = args.text

    if not text.strip():
        print("Error: No text provided")
        return

    # Load model
    model = load_classifier(args.checkpoint, num_labels)

    # Predict
    print("\nAnalyzing text...")
    pred_label, pred_prob, all_probs = predict_text(model, text, label_names)

    # Display results
    print("\n" + "="*50)
    print("PREDICTION RESULTS")
    print("="*50)
    print(f"\nMost likely source: {label_names[pred_label]}")
    print(f"Confidence: {pred_prob*100:.2f}%")

    print("\nAll probabilities:")
    print("-"*50)
    # Sort by probability
    sorted_indices = np.argsort(all_probs)[::-1]
    for idx in sorted_indices:
        bar_length = int(all_probs[idx] * 50)
        bar = "█" * bar_length
        print(f"{label_names[idx]:20s} {all_probs[idx]*100:6.2f}% {bar}")
    print("="*50)


if __name__ == "__main__":
    main()
