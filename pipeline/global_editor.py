import torch
import numpy as np
from PIL import Image
from diffusers import StableDiffusionControlNetImg2ImgPipeline, ControlNetModel


class GeometryAwareImg2ImgGenerator:
    def __init__(self, device="cuda", image_size=(768, 768)):
        self.device = device
        self.image_size = image_size

        # ── ControlNet models ──────────────────────────────────────
        controlnet_depth = ControlNetModel.from_pretrained(
            "lllyasviel/sd-controlnet-depth", torch_dtype=torch.float16
        )
        controlnet_line = ControlNetModel.from_pretrained(
            "lllyasviel/sd-controlnet-mlsd", torch_dtype=torch.float16
        )

        # ── Diffusion pipeline ─────────────────────────────────────
        self.pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            controlnet=[controlnet_depth, controlnet_line],
            torch_dtype=torch.float16,
        ).to(device)
        self.pipe.enable_xformers_memory_efficient_attention()

    def preprocess(self, image: Image.Image):
        return image.resize(self.image_size, Image.LANCZOS)

    def extract_controls(self, line_map, depth_map):
        depth_map = depth_map.resize(self.image_size)
        line_map = line_map.resize(self.image_size)
        return depth_map, line_map

    def generate(
        self,
        original_image: Image.Image,
        line_map: Image.Image,
        depth_map: Image.Image,
        prompt: str,
        negative_prompt: str = "blurry, distorted walls, wrong perspective, deformed",
        strength: float = 0.35,
        guidance_scale: float = 7.5,
        steps: int = 30):

        image = self.preprocess(original_image)
        depth_map, line_map = self.extract_controls(line_map, depth_map)

        output = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=image,
            control_image=[depth_map, line_map],
            controlnet_conditioning_scale=[0.8, 0.4],  # depth dominates
            strength=strength,
            guidance_scale=guidance_scale,
            num_inference_steps=steps,
        )

        return output.images[0]
