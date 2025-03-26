# 🌀 DLSF: Dual-Layer Synergistic Fusion for High-Fidelity Image Synthesis

> This repository provides the inference pipeline for **DLSF (Dual-Layer Synergistic Fusion)** — a novel framework that enhances image generation quality in **Stable Diffusion XL (SDXL)** through two complementary fusion strategies: **Adaptive Global Fusion (AGF)** and **Dynamic Spatial Fusion (DSF)**.

---

## 🌐 Project Links

- [📄 Paper]  |  [🔗 Project Page]  |  [🎞️ Video]  |  [💻 Code]

---

## ✨ Highlights

- 🔁 Seamlessly fuses **base** and **refined** latents using learnable fusion modules.
- 🧠 AGF & DSF modules optimize both semantic alignment and spatial detail.
- 🖼️ Enables high-resolution (1024×1024) image generation via SDXL decoding.
- ⚡ Built on PyTorch and HuggingFace Diffusers with FP16 support.
- 📈 Outperforms SDXL in FID, IS, and Recall on ImageNet benchmarks.

---

## 🧠 Method Summary

While standard SDXL pipelines run base and refiner models independently, DLSF introduces latent-space fusion:

- **AGF (Adaptive Global Fusion)** — aligns features across semantic levels using learnable attention weights.
- **DSF (Dynamic Spatial Fusion)** — applies spatial attention to enhance pixel-level detail.

The fused latent is decoded by SDXL’s VAE into the final high-resolution image.

---

## 📁 Project Structure

```
DLSF-Inference/
├── DLSF_module.py            # Full pipeline: fusion + decoding + inference
├── evaluator.py              # Evaluation script (FID, IS, sFID, etc.)
├── requirements.txt          # Python dependency list
└── README.md                 # Project documentation
```

---

## 🚀 Running Inference

Generate images with DLSF by running `DLSF_module.py`. The script performs the following steps:

1. Loads SDXL base and refiner models from HuggingFace.
2. Encodes the prompt to generate latent features.
3. Applies AGF or DSF to fuse base and refined latents.
4. Decodes the fused latent into a 1024×1024 image.
5. Applies optional watermarking and postprocessing.
6. Saves the result as a `.jpg` file.

---

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

(Optional for speed):

```bash
pip install xformers
```

---

### 2. Run the Inference Script

```bash
python DLSF_module.py
```

- Uses default prompt and `"DSF"` strategy.
- Output saved as `output.jpg`.

---

### 3. Customize in Python

You can also use the DLSF pipeline directly:

```python
from DLSF_module import run_dlsf_inference

image = run_dlsf_inference(
    prompt="a dragon-shaped hot air balloon flying over the Grand Canyon",
    fusion_type="AGF",  # or "DSF"
    device="cuda"       # or "cpu"
)
image.save("custom_output.jpg")
```

- `prompt`: Custom text-to-image description
- `fusion_type`: `"AGF"` (semantic) or `"DSF"` (detail)
- `device`: `"cuda"` recommended

---

### 4. Modify Defaults in Script

In `DLSF_module.py`, edit:

```python
prompt = "your custom prompt here"
fusion = "AGF"
```

Then re-run the script.

---

### 5. Internal Workflow

- Loads models via `StableDiffusionPipeline`
- Generates and fuses latents via AGF/DSF
- Decodes image using SDXL’s VAE
- Post-processes and saves output

---

## 🖼️ Output Samples

| Fusion | Prompt                                                     | Output                                      |
|--------|------------------------------------------------------------|---------------------------------------------|
| AGF    | *a futuristic cityscape at night*                          | <img src="image/example1.jpg" width="320"/> |
| DSF    | *a hot air balloon flying over the Grand Canyon at sunset*| <img src="image/example2.jpg" width="320"/> |

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

### 🔬 Ablation Study (512×512 with Refinement Step `/r`)

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
- GPU: NVIDIA A6000 or GPU with ≥24GB VRAM
- Recommended: `diffusers==0.24.0`, `xformers` (optional)

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

- Extending to video and 3D generation
- Applications in medical imaging, remote sensing
- Open-sourcing the training pipeline
