import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.pipelines.stable_diffusion_xl.pipeline_output import StableDiffusionXLPipelineOutput

# -------------------------------
# Fusion Modules
# -------------------------------

class AdaptiveGlobalFusion(nn.Module):
    def __init__(self, channel_dim):
        super().__init__()
        self.attention = nn.Conv2d(channel_dim * 2, 2, kernel_size=1)

    def forward(self, base_latent, refined_latent):
        x = torch.cat([base_latent, refined_latent], dim=1)
        logits = self.attention(x)
        weights = F.softmax(logits, dim=1)
        w_b = weights[:, 0:1, :, :]
        w_r = weights[:, 1:2, :, :]
        return w_b * base_latent + w_r * refined_latent


class DynamicSpatialFusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.spatial_attn = nn.Conv2d(2, 1, kernel_size=7, padding=3)

    def forward(self, base_latent, refined_latent):
        avg_pool = torch.mean(refined_latent, dim=1, keepdim=True)
        max_pool, _ = torch.max(base_latent, dim=1, keepdim=True)
        concat = torch.cat([avg_pool, max_pool], dim=1)
        attention = torch.sigmoid(self.spatial_attn(concat))
        return attention * refined_latent + (1 - attention) * base_latent

# -------------------------------
# Image Decoder Utility
# -------------------------------

def decode_image(latents, pipe, output_type="pil"):
    torch.set_grad_enabled(False)
    needs_upcasting = pipe.vae.dtype == torch.float16 and pipe.vae.config.force_upcast

    if needs_upcasting:
        pipe.upcast_vae()
        latents = latents.to(next(iter(pipe.vae.post_quant_conv.parameters())).dtype)
    elif latents.dtype != pipe.vae.dtype:
        if torch.backends.mps.is_available():
            pipe.vae = pipe.vae.to(latents.dtype)

    has_latents_mean = hasattr(pipe.vae.config, "latents_mean") and pipe.vae.config.latents_mean is not None
    has_latents_std = hasattr(pipe.vae.config, "latents_std") and pipe.vae.config.latents_std is not None

    if has_latents_mean and has_latents_std:
        latents_mean = torch.tensor(pipe.vae.config.latents_mean).view(1, 4, 1, 1).to(latents.device, latents.dtype)
        latents_std = torch.tensor(pipe.vae.config.latents_std).view(1, 4, 1, 1).to(latents.device, latents.dtype)
        latents = latents * latents_std / pipe.vae.config.scaling_factor + latents_mean
    else:
        latents = latents / pipe.vae.config.scaling_factor

    image = pipe.vae.decode(latents, return_dict=False)[0]

    if needs_upcasting:
        pipe.vae.to(dtype=torch.float16)

    if pipe.watermark is not None:
        image = pipe.watermark.apply_watermark(image)

    image = pipe.image_processor.postprocess(image.detach(), output_type=output_type)
    pipe.maybe_free_model_hooks()

    return StableDiffusionXLPipelineOutput(images=image)

# -------------------------------
# Complete DLSF Inference Pipeline
# -------------------------------

def run_dlsf_inference(prompt, fusion_type="AGF", device="cuda"):
    from diffusers import StableDiffusionPipeline

    base_model_id = "stabilityai/stable-diffusion-xl-base-1.0"
    refiner_model_id = "stabilityai/stable-diffusion-xl-refiner-1.0"

    base_pipeline = StableDiffusionPipeline.from_pretrained(
        base_model_id, torch_dtype=torch.float16, variant="fp16"
    ).to(device)

    refiner_pipeline = StableDiffusionPipeline.from_pretrained(
        refiner_model_id, torch_dtype=torch.float16, variant="fp16"
    ).to(device)

    with torch.autocast(device):
        base_output = base_pipeline(prompt, output_type="latent")
        base_latents = base_output["latents"]

        refiner_output = refiner_pipeline(prompt, output_type="latent")
        refined_latents = refiner_output["latents"]

    if fusion_type.upper() == "AGF":
        fusion_module = AdaptiveGlobalFusion(channel_dim=base_latents.shape[1])
    elif fusion_type.upper() == "DSF":
        fusion_module = DynamicSpatialFusion()
    else:
        raise ValueError("Invalid fusion type. Choose 'AGF' or 'DSF'.")

    fusion_module.to(device)
    with torch.no_grad():
        fused_latents = fusion_module(base_latents, refined_latents)

    final_image = decode_image(fused_latents, base_pipeline, output_type="pil").images[0]
    return final_image

if __name__ == "__main__":
    prompt = "a majestic lion in a surreal cyberpunk jungle"
    fusion = "DSF"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    image = run_dlsf_inference(prompt, fusion_type=fusion, device=device)
    image.save("output.jpg")
