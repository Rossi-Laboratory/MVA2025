# 🌀 DLSF: Dual-Layer Synergistic Fusion for High-Fidelity Image Synthesis

> This repository provides the inference pipeline for **DLSF (Dual-Layer Synergistic Fusion)**, a framework designed to improve image generation quality in **Stable Diffusion XL (SDXL)** models using two novel fusion strategies — **Adaptive Global Fusion (AGF)** and **Dynamic Spatial Fusion (DSF)**.

---

## 🌐 Project Links

- [📄 Paper]  |  [🔗 Project Page](https://rossi-laboratory.github.io/MVA2025/)  |  [🎞️ Video]  |  [💻 Code]

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

Follow the steps below to generate high-resolution images using DLSF:

### 1. Clone the repository and launch the notebook

```bash
git clone https://github.com/your-username/DLSF-Inference.git
cd DLSF-Inference
jupyter notebook
```

Open `inference.ipynb` in Jupyter Notebook.

---

### 2. Load models and select fusion strategy

The notebook loads:
- The **SDXL base model**: `stable-diffusion-xl-base-1.0`
- The **SDXL refiner model**: `stable-diffusion-xl-refiner-1.0`

Both models are fetched from HuggingFace using `from_pretrained()` and run in FP16 for efficiency.

Choose one of the two fusion strategies:
```python
fusion_type = "AGF"  # or "DSF"
```

---

### 3. Enter your custom prompt

Set your desired prompt (text description of the image to generate):

```python
prompt = "a majestic lion in a surreal cyberpunk jungle"
```

The prompt will be encoded via SDXL’s dual-text encoder to guide image generation.

---

### 4. Run the inference pipeline

The pipeline proceeds through the following steps:
- Generate **base latent** from prompt
- Refine it using the **refiner model**
- Fuse base and refined latents using the selected **AGF or DSF module**
- Decode the fused latent into a 1024×1024 image using SDXL’s VAE

The final image will be displayed and can also be saved locally.

---

### 5. Saving results (optional)

You can save the output image using:

```python
image.save("output.jpg")
```

---

This process takes ~10 seconds per image on an NVIDIA A6000. You may increase batch size or parallelism if GPU memory allows.


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
