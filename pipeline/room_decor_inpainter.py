import torch
import numpy as np
from PIL import Image
from diffusers import StableDiffusionInpaintPipeline


class DiffusionInpainter:

    def __init__(self, model_id="runwayml/stable-diffusion-inpainting", device="cuda"):

        self.device = device

        self.pipe = StableDiffusionInpaintPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16
        ).to(device)

    def preprocess_mask(self, mask):

        """
        Convert boolean or tensor mask → PIL mask
        Diffusion requires:
        white = region to edit
        black = keep original
        """

        if isinstance(mask, torch.Tensor):
            mask = mask.cpu().numpy()

        #mask = mask.astype(np.uint8) * 255
        #cv2.imwrite('in-mask.png', mask)

        mask = Image.fromarray(mask)
        mask.save('in-mask.png')
        return mask

    def inpaint(
        self,
        image,
        mask,
        prompt,
        negative_prompt="blurry, distorted, bad perspective",
        steps=30,
        guidance=7.5
    ):

        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)

        mask = self.preprocess_mask(mask)

        #image = image.convert("RGB")
        mask = mask.convert("L")
        #generator = torch.Generator(device="cuda").manual_seed(0)
        result = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=image,
            mask_image=mask, strength=0.99,
            num_inference_steps=steps,
            guidance_scale=guidance, #generator=generator,
        ).images[0]

        return result

    def inpaint_batch(
        self,
        images,
        masks,
        prompts,
        negative_prompt="blurry, distorted, bad perspective",
        steps=30,
        guidance=7.5
    ):
        """
        images: list of PIL Images
        masks: list of PIL masks (same size as corresponding images)
        prompts: list of strings
        """
        # Convert all masks to L mode
        masks = [mask.convert("L") for mask in masks]

        results = self.pipe(
            prompt=prompts,
            negative_prompt=[negative_prompt]*len(prompts),
            image=images,
            mask_image=masks,
            strength=0.99,
            num_inference_steps=steps,
            guidance_scale=guidance
        ).images

        return results
