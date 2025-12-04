import argparse
import json
import os
import tempfile
import torch
import numpy as np
import gc
from transformers import AutoConfig, AutoModel, AutoTokenizer
from peft import PeftModel
from llm2vec import LLM2Vec


def load_classifier(checkpoint_path, mode="LOW_BANDWIDTH", num_labels=5):
    """
    Robust classifier loader with Dual-Pathway support.
    mode="FAST_FUSED": Loads pre-merged weights (Fast startup, High disk usage)
    mode="LOW_BANDWIDTH": Merges adapters at runtime (Slow startup, Low disk usage)
    """
    print(f"Loading classifier in {mode} mode...")
    torch.cuda.empty_cache()
    gc.collect()

    # Common Resource: The Base Model ID (Used for Config/Tokenizer in both modes)
    base_model_id = "McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp"
    tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)

    # Memory configuration
    max_memory = {0: "14GiB", "cpu": "30GiB"}

    # =========================================================
    # PATHWAY A: FAST / FUSED (Hybrid Load with Safety)
    # =========================================================
    if mode == "FAST_FUSED":
        print("⚡ Route: Loading pre-merged weights...")

        # 1. Load Configuration from Base (Critical Fix for missing config.json)
        config = AutoConfig.from_pretrained(base_model_id, trust_remote_code=True)

        # 2. Load Weights from Local Folder with SAFE device map
        # NOTE: Using single GPU strategy to avoid residual connection device mismatch
        device_map = {"": 0}  # Force everything to GPU 0

        model = AutoModel.from_pretrained(
            checkpoint_path,
            config=config,
            torch_dtype=torch.bfloat16,
            device_map=device_map,
            max_memory=max_memory,
            offload_folder="./offload",
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )

        print("✓ Pre-merged model loaded")
        torch.cuda.empty_cache()
        gc.collect()

    # =========================================================
    # PATHWAY B: LOW BANDWIDTH / ADAPTER (Original Logic)
    # =========================================================
    else:
        print("🔧 Route: Building from adapters...")

        # 1. Load Base Model
        print("Loading base model...")
        config = AutoConfig.from_pretrained(base_model_id, trust_remote_code=True)

        device_map = {"": 0}
        model = AutoModel.from_pretrained(
            base_model_id,
            config=config,
            torch_dtype=torch.bfloat16,
            device_map=device_map,
            max_memory=max_memory,
            offload_folder="./offload",
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )

        print("✓ Base model loaded")
        torch.cuda.empty_cache()

        # 2. Load and merge first adapter
        print("Loading first adapter...")
        model = PeftModel.from_pretrained(
            model,
            base_model_id,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )

        # Move to CPU for merging to avoid OOM
        print("✓ Moving to CPU for merging...")
        model = model.cpu()
        torch.cuda.empty_cache()
        gc.collect()

        print("✓ Merging first adapter...")
        model = model.merge_and_unload()

        # Move back to GPU with CPU offload
        print("✓ Moving merged model back to GPU...")
        model = model.to(torch.bfloat16)

        # Re-dispatch to GPU 0 (with CPU offload for layers that don't fit)
        from accelerate import dispatch_model, infer_auto_device_map
        device_map_merged = infer_auto_device_map(
            model,
            max_memory=max_memory,
            no_split_module_classes=["LlamaDecoderLayer"]  # Keep decoder layers intact
        )
        model = dispatch_model(model, device_map=device_map_merged, offload_dir="./offload")

        torch.cuda.empty_cache()
        gc.collect()

        # 3. Load supervised adapter
        print("✓ Loading supervised adapter...")
        model = PeftModel.from_pretrained(
            model,
            f"{base_model_id}-supervised",
            is_trainable=True,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )

        torch.cuda.empty_cache()

    # =========================================================
    # COMMON: LLM2Vec & Classification Head
    # =========================================================
    model = LLM2Vec(model, tokenizer, pooling_mode="mean", max_length=512)

    # Initialize Head
    hidden_size = list(model.modules())[-1].weight.shape[0]
    model.head = torch.nn.Linear(hidden_size, num_labels, dtype=torch.bfloat16)

    # Load Head Weights with dynamic device detection
    head_file = os.path.join(checkpoint_path, "head.pt")
    if not os.path.exists(head_file):
        raise FileNotFoundError(f"Classification head not found at {head_file}")

    try:
        target_device = next(model.parameters()).device
    except:
        target_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model.head.load_state_dict(torch.load(head_file, map_location=target_device))
    model.head = model.head.to(target_device)
    print(f"✓ Loaded classification head from {head_file}")

    model.eval()

    # Final cleanup
    torch.cuda.empty_cache()
    gc.collect()

    # Show memory usage
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / 1024**3
            reserved = torch.cuda.memory_reserved(i) / 1024**3
            if i == 0 or allocated > 0:
                print(f"✓ GPU {i}: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
    else:
        print("✓ Classifier loaded on CPU")

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
