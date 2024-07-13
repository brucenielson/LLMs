# https://medium.com/@demstalfer/quick-guide-to-using-the-new-stable-diffusion-xl-for-code-enthusiasts-bd71136dd794
import random
import sys
import torch
import os
import datetime
from diffusers import DiffusionPipeline
from accelerate import Accelerator
from typing import List, Tuple, Union


class StableDiffusionXLPipeline:
    def __init__(self, use_refiner: bool = True, height: int = None, width: int = None,
                 guidance_scale: float = 5.0, num_images_per_prompt: int = 1):
        self.use_refiner = use_refiner
        self.height = height
        self.width = width
        self.guidance_scale = guidance_scale
        self.num_images_per_prompt = num_images_per_prompt
        self.output_dir = 'output_images'
        self.torch_dtype = None
        self.accelerator = None
        self.device = None
        self.refiner = None
        self.pipe = None
        self.setup_pipeline()

    def setup_pipeline(self):
        self.torch_dtype = torch.float16
        self.accelerator = Accelerator()
        self.device = self.accelerator.device

        self.pipe = DiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=self.torch_dtype,
            use_safetensors=True,
            variant="fp16"
        )

        if self.use_refiner:
            self.refiner = DiffusionPipeline.from_pretrained(
                "stabilityai/stable-diffusion-xl-refiner-1.0",
                text_encoder_2=self.pipe.text_encoder_2,
                vae=self.pipe.vae,
                torch_dtype=self.torch_dtype,
                use_safetensors=True,
                variant="fp16"
            ).to(self.device)
            self.pipe.enable_model_cpu_offload()
        else:
            self.pipe = self.pipe.to(self.device)

    def process_prompts(self, prompts: List[Union[str, Tuple[str, ...]]]):
        for prompt_tuple in prompts:
            self._process_single_prompt(prompt_tuple)

    def _process_single_prompt(self, prompt_tuple: Union[str, Tuple[str, ...]]):
        prompt, prompt_2, negative_prompt, negative_prompt_2 = (
            StableDiffusionXLPipeline._parse_prompt_tuple(prompt_tuple))
        seed = random.randint(0, sys.maxsize)
        generator = torch.Generator(self.device).manual_seed(seed)

        print(f"Prompt:\t{prompt}")
        print(f"Seed:\t{seed}")

        images = self._generate_images(prompt, prompt_2, negative_prompt, negative_prompt_2, generator)
        self._save_images(images, prompt)

    from typing import Union, Tuple, Optional

    @staticmethod
    def _parse_prompt_tuple(prompt_tuple: Union[str, Tuple[str, ...]]) \
            -> Tuple[str, Optional[str], Optional[str], Optional[str]]:
        if isinstance(prompt_tuple, str):
            return prompt_tuple, None, None, None
        return (prompt_tuple[0],
                prompt_tuple[1] if len(prompt_tuple) > 1 else None,
                prompt_tuple[2] if len(prompt_tuple) > 2 else None,
                prompt_tuple[3] if len(prompt_tuple) > 3 else None)

    def _generate_images(self, prompt, prompt_2, negative_prompt, negative_prompt_2, generator):
        output_type = "latent" if self.use_refiner else "pil"
        images = self.pipe(
            prompt=prompt,
            prompt_2=prompt_2,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt_2,
            output_type=output_type,
            generator=generator,
            height=self.height,
            width=self.width,
            guidance_scale=self.guidance_scale,
            num_images_per_prompt=self.num_images_per_prompt
        ).images

        if self.use_refiner:
            images = self.refiner(prompt=prompt, image=images).images

        return images

    def _save_images(self, images, prompt):
        os.makedirs(self.output_dir, exist_ok=True)
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

        for i, image in enumerate(images):
            image_path = os.path.join(self.output_dir, f"output_{timestamp}_{i}.jpg")
            print(f"Saving image to {image_path}")
            image.save(image_path)

        print(f"Image complete. {prompt}")


# Usage
if __name__ == "__main__":
    prompts_to_process = [
        "A fairy princess and her majestic dragon. Photorealistic."
    ]

    pipeline = StableDiffusionXLPipeline(use_refiner=True, num_images_per_prompt=1)
    pipeline.process_prompts(prompts_to_process)