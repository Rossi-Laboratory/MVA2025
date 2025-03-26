# 🌀 DLSF: Dual-Layer Synergistic Fusion for High-Fidelity Image Synthesis

> This repository provides the inference pipeline for **DLSF (Dual-Layer Synergistic Fusion)**, a framework designed to improve image generation quality in **Stable Diffusion XL (SDXL)** models using two novel fusion strategies — **Adaptive Global Fusion (AGF)** and **Dynamic Spatial Fusion (DSF)**.

---

## 🌐 Project Links

- [📄 Paper]  |  [🔗 Project Page]  |  [🎞️ Video]  |  [💻 Code]

---

## ✨ Highlights

- 🔁 Combines **base** and **refined** latents through learnable fusion modules.
- 🧠 AGF & DSF modules preserve both global semantics and local details.
- 🎨 Supports text-to-image and multi-view generation at 1024×1024 resolution.
- ⚡ FP16-optimized inference using HuggingFace Diffusers + PyTorch.
- 📈 Improved FID, IS, and Recall over baseline SDXL on ImageNet.

---

## 🧠 Method Summary

Standard SDXL pipelines sequentially apply a base and refiner model but fuse their latents suboptimally.

**DLSF** solves this by:

- 🌀 **AGF (Adaptive Global Fusion)** — Attention-based fusion for aligning hierarchical features.
- 🧭 **DSF (Dynamic Spatial Fusion)** — Spatial attention to emphasize high-frequency details.

The resulting latent is decoded by SDXL’s VAE into a final high-resolution image.

---

## 📁 Project Structure

```
DLSF-Inference/
├── inference.ipynb         # Main inference demo (Jupyter Notebook)
├── fusion_modules.py       # AGF and DSF fusion logic
├── requirements.txt        # Python dependencies
└── evaluator.py            # Evaluation script for metrics (FID, IS, etc.)
```

---

## ⚙️ Installation

Install required dependencies:

```bash
pip install -r requirements.txt
```

Optionally, install `xformers` for faster inference:

```bash
pip install xformers
```

---
## 🚀 Running Inference

You can generate images using the complete DLSF pipeline via the script `DLSF_module.py`.  
This script performs the following steps:

1. Loads SDXL base and refiner models from HuggingFace.
2. Generates base and refined latent features from your text prompt.
3. Applies the AGF or DSF fusion module to combine latents.
4. Decodes the fused latent into a 1024×1024 image using the VAE.
5. Applies postprocessing and watermarking (if applicable).
6. Saves the final image as a `.jpg` file.

---

### 1. Install Dependencies

Make sure you have Python 3.9+ and install the required packages:

```bash
pip install -r requirements.txt
```

(Optional) For faster inference on compatible GPUs:

```bash
pip install xformers
```

---

### 2. Run Inference from Script

You can run the default example using:

```bash
python DLSF_module.py
```

- This uses a default prompt defined in the script.
- The selected fusion strategy is `"DSF"` by default.
- The result will be saved as `output.jpg` in the current directory.

---

### 3. Customize in Python

You can also use the DLSF inference pipeline directly in your own Python scripts:

```python
from DLSF_module import run_dlsf_inference

image = run_dlsf_inference(
    prompt="a dragon-shaped hot air balloon flying over the Grand Canyon",
    fusion_type="AGF",  # or "DSF"
    device="cuda"       # or "cpu"
)
image.save("custom_output.jpg")
```

- `prompt`: A custom text description of the image you want.
- `fusion_type`: Choose between `"AGF"` for semantic alignment or `"DSF"` for spatial detail.
- `device`: `"cuda"` for GPU (recommended), or `"cpu"` for testing.

The returned `image` is a `PIL.Image` object that you can view, save, or post-process.

---

### 4. Modify Prompt and Fusion Type in Script

In `DLSF_module.py`, you can directly change these two lines to test different prompts or strategies:

```python
prompt = "your custom prompt here"
fusion = "AGF"  # or "DSF"
```

Save and rerun the script to regenerate the image.

---

### 5. What Happens Internally

- The script loads SDXL base + refiner models with `StableDiffusionPipeline`.
- `run_dlsf_inference()` generates two latent outputs, fuses them, and decodes to an image.
- `decode_image()` handles unscaling, watermarking, and output conversion.







---

## 🖼️ Output Samples

DLSF demonstrates strong semantic alignment and texture fidelity. Below are sample outputs using AGF and DSF fusion strategies:

| Fusion | Prompt                                                         | Output                                        |
|--------|----------------------------------------------------------------|-----------------------------------------------|
| AGF    | *a futuristic cityscape at night*                              | <img src="image/example1.jpg" width="320"/>   |
| DSF    | *a hot air balloon flying over the Grand Canyon at sunset*     | <img src="image/example2.jpg" width="320"/>   |



---


## 📊 Performance on ImageNet (Class-Conditional)

### 🔹 256×256 Resolution

| Method | FID ↓ | sFID ↓ | IS ↑    | Precision ↑ | Recall ↑ |
|--------|-------|--------|--------|--------------|-----------|
| SDXL   | 20.16 | 48.98  | 219.74 | 0.860        | 0.350     |
| AGF    | 18.79 | 47.64  | 230.43 | 0.870        | 0.390     |
| DSF    | 18.89 | 48.21  | 232.04 | 0.870        | 0.390     |

### 🔹 512×512 Resolution

| Method | FID ↓ | sFID ↓ | IS ↑    | Precision ↑ | Recall ↑ |
|--------|-------|--------|--------|--------------|-----------|
| SDXL   | 19.65 | 50.54  | 234.75 | 0.860        | 0.350     |
| AGF    | 18.70 | 49.77  | 243.48 | 0.852        | 0.381     |
| DSF    | 18.70 | 50.22  | 243.62 | 0.854        | 0.383     |

### 🔬 Ablation Study (512×512 with Additional Refinement Step `/r`)

| Method     | FID ↓ | sFID ↓ | IS ↑    | Precision ↑ | Recall ↑ |
|------------|-------|--------|--------|--------------|-----------|
| AGF        | 18.70 | 49.77  | 243.48 | 0.852        | 0.381     |
| AGF /r     | 20.02 | 53.94  | 215.06 | 0.851        | 0.363     |
| DSF        | 18.70 | 50.22  | 243.62 | 0.854        | 0.383     |
| DSF /r     | 19.89 | 54.28  | 218.36 | 0.853        | 0.383     |


---

## 💻 Environment

- Python 3.9+
- PyTorch ≥ 2.0
- GPU: NVIDIA A6000 or ≥24GB VRAM recommended
- Tested with: `diffusers==0.24.0`

---

## 📚 Citation

```bibtex
@inproceedings{DLSF2025,
  title={DLSF: Dual-Layer Synergistic Fusion for High-Fidelity Image Synthesis},
  year={2025}
}
```

---

## 🔮 Future Work

- Video and 3D content generation
- Applications in medical and industrial domains
- Releasing training pipeline for community fine-tuning
