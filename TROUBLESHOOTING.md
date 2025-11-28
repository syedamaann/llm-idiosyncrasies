# Troubleshooting Guide

## Common Issues and Solutions

### 1. ModuleNotFoundError: transformers.modeling_layers

**Error:**
```
ModuleNotFoundError: No module named 'transformers.modeling_layers'
```

**Solution:**
This is a version compatibility issue between `transformers` and `peft`. Install specific compatible versions:

```bash
pip install transformers==4.46.3 peft==0.13.2
pip install llm2vec==0.2.3
```

**In Google Colab:** The notebook has been updated with the correct versions. Just re-run the first cell.

---

### 2. Out of Memory (OOM) Error

**Error:**
```
CUDA out of memory
```

**Solutions:**

**For Google Colab:**
- Make sure you selected **T4 GPU** (not CPU)
- Runtime → Change runtime type → Hardware accelerator → GPU
- Restart runtime and try again

**For local machine:**
- You need at least 16GB GPU memory
- Try using a cloud service instead (see CLOUD_OPTIONS.md)

---

### 3. Slow Download Speed

**Issue:** Downloading the 16GB model takes forever

**Solutions:**
- Use a stable internet connection
- The download only happens once per session
- For repeated use, consider:
  - Hugging Face Spaces (model stays loaded)
  - Local setup if you have the hardware

---

### 4. Runtime Disconnected (Colab)

**Issue:** Google Colab disconnects and you lose progress

**Solutions:**
- Keep the browser tab active
- Run a keep-alive script:
```javascript
// In browser console (F12)
setInterval(() => {
  document.querySelector("colab-connect-button").click()
}, 60000)
```
- Or use Hugging Face Spaces for permanent deployment

---

### 5. "No module named 'llm2vec'"

**Solution:**
```bash
pip install llm2vec==0.2.3
```

Make sure you run the installation cell first before importing.

---

### 6. Model Download Fails

**Error:**
```
Error downloading from Hugging Face
```

**Solutions:**
1. Check internet connection
2. Try manual download:
```bash
huggingface-cli login  # if needed
huggingface-cli download Yida/classifier_chat --local-dir ./classifier_chat
```
3. Check Hugging Face status: https://status.huggingface.co

---

### 7. Prediction Takes Too Long

**Issue:** Classification is very slow

**Check:**
- Are you using GPU? Run this to verify:
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
```

**Solutions:**
- **If CPU:** Make sure GPU is enabled in Colab runtime settings
- **If GPU but slow:** First prediction is always slower (model initialization)

---

### 8. AttributeError with Model

**Error:**
```
AttributeError: 'LLM2Vec' object has no attribute 'head'
```

**Solution:**
Make sure you load the classifier completely, including the classification head:
```python
model.head.load_state_dict(torch.load(head_path, ...))
```

The notebook handles this automatically. If using custom code, check `classify_text.py` for reference.

---

### 9. Input Text Not Detected

**Issue:** Interactive input doesn't work in Colab

**Solution:**
Use the direct text variable instead:
```python
# Instead of input()
my_text = """
Your text here
"""
predict_text(model, my_text)
```

---

### 10. Package Version Conflicts

**Error:**
```
ERROR: pip's dependency resolver does not currently take into account all the packages that are installed
```

**Solution:**
Install in this exact order:
```bash
pip install transformers==4.46.3
pip install peft==0.13.2
pip install llm2vec==0.2.3
pip install huggingface-hub
```

Or use a fresh environment:
```bash
# Restart Colab runtime: Runtime → Restart runtime
# Then run installation cell again
```

---

## Still Having Issues?

1. **Check package versions:**
```python
import transformers
import peft
import llm2vec
print(f"transformers: {transformers.__version__}")
print(f"peft: {peft.__version__}")
print(f"llm2vec: {llm2vec.__version__}")
```

Expected versions:
- transformers: 4.46.3
- peft: 0.13.2
- llm2vec: 0.2.3

2. **Restart runtime:**
   - Colab: Runtime → Restart runtime
   - Local: Restart Python kernel

3. **Try a different cloud provider:**
   - See `CLOUD_OPTIONS.md` for alternatives to Colab

4. **Check the original repository:**
   - https://github.com/eric-mingjie/llm-idiosyncrasies
