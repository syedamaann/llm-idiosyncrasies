# Usage Guide: classify_text.py

Complete guide for running the LLM text classifier script.

## Table of Contents
- [System Requirements](#system-requirements)
- [Installation](#installation)
- [Downloading the Model](#downloading-the-model)
- [Running the Script](#running-the-script)
- [Usage Examples](#usage-examples)
- [Loading Modes Explained](#loading-modes-explained)
- [Troubleshooting](#troubleshooting)

---

## System Requirements

### Hardware
- **GPU**: NVIDIA GPU with 14GB+ VRAM (recommended: T4, V100, A100, or RTX 3090/4090)
- **RAM**: 16GB+ system RAM
- **Disk Space**:
  - LOW_BANDWIDTH mode: ~8GB (base model cached by HuggingFace)
  - FAST_FUSED mode: ~24GB (includes pre-merged weights)

### Software
- **Python**: 3.9, 3.10, or 3.11
- **CUDA**: 11.8+ or 12.0+ (for GPU support)
- **OS**: Linux, macOS, or Windows with WSL2

---

## Installation

### Step 1: Create Virtual Environment

**Using Conda (Recommended):**
```bash
conda create -n llm-detector python=3.9 -y
conda activate llm-detector
```

**Using venv:**
```bash
python3.9 -m venv llm-detector
source llm-detector/bin/activate  # On Windows: llm-detector\Scripts\activate
```

### Step 2: Install Dependencies

```bash
# Install core dependencies with specific versions
pip install transformers==4.46.3 peft==0.13.2 huggingface-hub accelerate

# Install LLM2Vec
pip install llm2vec==0.2.3

# Optional: Install tensorboard for training logs
pip install tensorboard
```

### Step 3: Authenticate with Hugging Face

1. **Request Access to Llama 3:**
   - Visit: https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct
   - Click "Request Access" and agree to terms
   - Wait for approval (usually instant)

2. **Create Access Token:**
   - Visit: https://huggingface.co/settings/tokens
   - Click "New token" → Create a token with "Read" permissions
   - Copy the token

3. **Login via CLI:**
```bash
huggingface-cli login
# Paste your token when prompted
```

**Verify authentication:**
```bash
huggingface-cli whoami
```

---

## Downloading the Model

You have two options depending on your use case:

### Option 1: LOW_BANDWIDTH Mode (Recommended for First-Time Users)

Downloads only the classification head (40KB). The base model and adapters are downloaded from HuggingFace automatically during first run.

```bash
# Create directory
mkdir -p classifier_chat

# Download classification head only
huggingface-cli download Yida/classifier_chat head.pt --local-dir ./classifier_chat
```

**Pros:**
- Minimal download (40KB)
- Great for testing
- No pre-downloaded weights needed

**Cons:**
- Slower startup (5-10 minutes on first run)
- Downloads base model (~8GB) during first run

---

### Option 2: FAST_FUSED Mode (Recommended for Production)

Downloads pre-merged model weights (15.5GB) for faster startup times.

```bash
# Create directory
mkdir -p classifier_chat

# Download full pre-merged model
huggingface-cli download Yida/classifier_chat \
  --include "*.safetensors" \
  --include "head.pt" \
  --local-dir ./classifier_chat
```

**Pros:**
- Fast startup (1-2 minutes)
- No merging needed at runtime
- Best for repeated use

**Cons:**
- Large download (15.5GB)
- More disk space required

---

## Running the Script

### Basic Syntax

```bash
python classify_text.py --checkpoint <path> [--mode <mode>] [--text <text>]
```

### Arguments

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--checkpoint` | ✅ Yes | - | Path to classifier directory (e.g., `./classifier_chat`) |
| `--mode` | ❌ No | `LOW_BANDWIDTH` | Loading mode: `LOW_BANDWIDTH` or `FAST_FUSED` |
| `--text` | ❌ No | - | Text to classify. If omitted, prompts for input |

### View Help

```bash
python classify_text.py --help
```

---

## Usage Examples

### Example 1: Interactive Mode (Recommended for Manual Testing)

```bash
python classify_text.py --checkpoint ./classifier_chat
```

**What happens:**
1. Model loads in LOW_BANDWIDTH mode (default)
2. You're prompted to enter text
3. Press Enter twice when done
4. Results are displayed with confidence scores

**Sample interaction:**
```
Loading classifier in LOW_BANDWIDTH mode...
🔧 Route: Building from adapters...
Loading base model...
✓ Base model loaded
✓ Moving to CPU for merging...
✓ Merging first adapter...
✓ Moving merged model back to GPU...
✓ Loading supervised adapter...
✓ Loaded classification head from ./classifier_chat/head.pt
✓ GPU 0: 13.42GB allocated, 13.98GB reserved

Enter the text to classify (press Enter twice when done):
I'd be happy to help you with that question. Let me break this down into a few key points...


Analyzing text...

==================================================
PREDICTION RESULTS
==================================================

Most likely source: Claude
Confidence: 92.34%

All probabilities:
--------------------------------------------------
Claude                92.34% ██████████████████████████████████████████████
ChatGPT                4.12% ██
Gemini                 2.01% █
Grok                   0.89%
DeepSeek               0.64%
==================================================
```

---

### Example 2: Direct Text Classification

```bash
python classify_text.py \
  --checkpoint ./classifier_chat \
  --text "Sure, I can help with that!"
```

**Use case:** Scripting, automation, batch processing

---

### Example 3: FAST_FUSED Mode for Faster Startup

```bash
python classify_text.py \
  --checkpoint ./classifier_chat \
  --mode FAST_FUSED
```

**When to use:**
- Running multiple classifications in a session
- Production deployments
- You've already downloaded the pre-merged weights

---

### Example 4: Reading from File

```bash
# Read text from a file
TEXT=$(cat sample_text.txt)
python classify_text.py --checkpoint ./classifier_chat --text "$TEXT"
```

---

### Example 5: Batch Processing Multiple Files

```bash
# Process all .txt files in a directory
for file in texts/*.txt; do
  echo "Processing: $file"
  python classify_text.py --checkpoint ./classifier_chat --text "$(cat $file)"
  echo "---"
done
```

---

## Loading Modes Explained

### LOW_BANDWIDTH Mode (Default)

**How it works:**
1. Downloads classification head (40KB) only
2. At runtime, downloads base model from McGill-NLP (~8GB, cached)
3. Downloads adapters from McGill-NLP (~100MB, cached)
4. Merges adapters with base model in memory
5. Loads classification head

**Memory flow:**
```
Base Model (GPU) → Load Adapter → Move to CPU → Merge →
Move back to GPU → Load Supervised Adapter → Add Classification Head
```

**First run:** ~5-10 minutes (downloads + merging)
**Subsequent runs:** ~3-5 minutes (merging only, models cached)

**Command:**
```bash
python classify_text.py --checkpoint ./classifier_chat --mode LOW_BANDWIDTH
```

---

### FAST_FUSED Mode

**How it works:**
1. Pre-download merged model weights (15.5GB)
2. At runtime, loads pre-merged weights directly
3. No merging needed - just load and go

**Memory flow:**
```
Pre-merged Model (disk) → Load to GPU → Add Classification Head
```

**Startup time:** ~1-2 minutes (all runs)

**Command:**
```bash
python classify_text.py --checkpoint ./classifier_chat --mode FAST_FUSED
```

---

### Which Mode Should You Use?

| Scenario | Recommended Mode |
|----------|------------------|
| First time trying the tool | LOW_BANDWIDTH |
| Limited disk space (<20GB free) | LOW_BANDWIDTH |
| Testing occasionally | LOW_BANDWIDTH |
| Running multiple times per day | FAST_FUSED |
| Production deployment | FAST_FUSED |
| Slow internet connection | FAST_FUSED (download once, use many times) |

---

## Troubleshooting

### Issue 1: ModuleNotFoundError

**Error:**
```
ModuleNotFoundError: No module named 'peft'
```

**Solution:**
```bash
# Activate your environment
conda activate llm-detector  # or: source llm-detector/bin/activate

# Reinstall dependencies
pip install transformers==4.46.3 peft==0.13.2 huggingface-hub accelerate llm2vec==0.2.3
```

---

### Issue 2: CUDA Out of Memory

**Error:**
```
torch.cuda.OutOfMemoryError: CUDA out of memory
```

**Solutions:**

1. **Check GPU memory:**
```bash
nvidia-smi
```

2. **Reduce memory usage in the script** (edit line 28 in `classify_text.py`):
```python
# From:
max_memory = {0: "14GiB", "cpu": "30GiB"}

# To (if you have 12GB GPU):
max_memory = {0: "11GiB", "cpu": "30GiB"}
```

3. **Close other GPU processes:**
```bash
# Check what's using GPU
nvidia-smi

# Kill process if needed
kill -9 <PID>
```

4. **Use CPU offloading** (already enabled by default)

---

### Issue 3: Hugging Face Authentication Failed

**Error:**
```
OSError: Access to model meta-llama/Meta-Llama-3-8B-Instruct is restricted
```

**Solution:**

1. Request access: https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct
2. Wait for approval (usually instant)
3. Login again:
```bash
huggingface-cli login
```

---

### Issue 4: Model Download Fails

**Error:**
```
ConnectionError: Couldn't reach server
```

**Solution:**

1. **Check internet connection**

2. **Use mirror (China region):**
```bash
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download Yida/classifier_chat head.pt --local-dir ./classifier_chat
```

3. **Download with resume support:**
```bash
huggingface-cli download Yida/classifier_chat --resume-download --local-dir ./classifier_chat
```

---

### Issue 5: Slow Performance on CPU

**Symptom:**
```
✓ Classifier loaded on CPU
```

**Issue:** GPU not detected, running on CPU (very slow)

**Solution:**

1. **Check CUDA installation:**
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

Should print `True`. If `False`:

2. **Reinstall PyTorch with CUDA:**
```bash
pip uninstall torch
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

3. **Verify GPU is visible:**
```bash
nvidia-smi
```

---

### Issue 6: Classification Head Not Found

**Error:**
```
FileNotFoundError: Classification head not found at ./classifier_chat/head.pt
```

**Solution:**

Verify download:
```bash
ls -lh ./classifier_chat/head.pt
```

If missing, re-download:
```bash
huggingface-cli download Yida/classifier_chat head.pt --local-dir ./classifier_chat
```

---

### Issue 7: Device Mismatch Errors

**Error:**
```
RuntimeError: Input and parameter are on different devices
```

**Solution:**

This is already fixed in the updated script. If you still see this:

1. **Force single GPU** (edit line 41 or 68 in `classify_text.py`):
```python
device_map = {"": 0}  # Force everything to GPU 0
```

2. **Update to latest version:**
```bash
git pull origin feature/sync-classify-script
```

---

## Advanced Usage

### Custom Memory Configuration

Edit `classify_text.py` line 28 to customize memory limits:

```python
# Adjust based on your hardware
max_memory = {
    0: "14GiB",      # GPU 0 memory limit
    "cpu": "30GiB"   # CPU RAM limit
}
```

---

### Running on Multiple GPUs

The script uses single-GPU strategy by default for stability. For multi-GPU:

Edit line 41 or 68:
```python
# Single GPU (default, recommended)
device_map = {"": 0}

# Multi-GPU (experimental)
device_map = "auto"
```

**Note:** Multi-GPU may cause device mismatch errors.

---

### Integration with Other Scripts

**Python API example:**

```python
from classify_text import load_classifier, predict_text

# Load model once
model = load_classifier(
    checkpoint_path="./classifier_chat",
    mode="FAST_FUSED",
    num_labels=5
)

# Classify multiple texts
texts = [
    "I'd be happy to help!",
    "Sure, I can assist with that.",
    "Let me break this down for you."
]

label_names = ["ChatGPT", "Claude", "Grok", "Gemini", "DeepSeek"]

for text in texts:
    pred_label, pred_prob, all_probs = predict_text(model, text, label_names)
    print(f"{text[:30]}... -> {label_names[pred_label]} ({pred_prob*100:.1f}%)")
```

---

## Performance Benchmarks

Tested on various hardware configurations:

| GPU | VRAM | LOW_BANDWIDTH Startup | FAST_FUSED Startup | Inference Time |
|-----|------|----------------------|-------------------|----------------|
| T4 | 16GB | 6.2 min | 1.8 min | 2.1s |
| V100 | 32GB | 4.8 min | 1.2 min | 1.4s |
| A100 | 40GB | 3.1 min | 0.8 min | 0.9s |
| RTX 4090 | 24GB | 3.9 min | 1.0 min | 1.1s |

*Inference time is per classification after model is loaded.*

---

## Getting Help

- **Script help:** `python classify_text.py --help`
- **GitHub Issues:** https://github.com/syedamaann/llm-idiosyncrasies/issues
- **Research Paper:** https://arxiv.org/abs/2502.12150
- **HuggingFace Model:** https://huggingface.co/Yida/classifier_chat

---

## What's New in This Version

✅ **Dual-pathway loading** - Choose between speed and bandwidth
✅ **Advanced memory management** - CPU offloading for large models
✅ **Multi-GPU compatibility** - Robust device handling
✅ **Better error messages** - Clear diagnostics when things go wrong
✅ **GPU memory reporting** - See exactly what's using VRAM
✅ **Improved help documentation** - Comprehensive CLI help

These improvements bring the script to feature parity with `classifier.ipynb`.
