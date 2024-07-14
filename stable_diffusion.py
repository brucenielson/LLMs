import random
import sys
import torch
import os
import datetime
from diffusers import DiffusionPipeline
from accelerate import Accelerator
from typing import List, Union, Dict, Optional


class StableDiffusionXLPipeline:
    def __init__(self, use_refiner: bool = True, height: int = 768, width: int = 768,
                 guidance_scale: float = 5.0, num_images_per_prompt: int = 1, output_dir: str = 'output_images'):
        self._use_refiner = use_refiner
        self._height = height
        self._width = width
        self._guidance_scale = guidance_scale
        self._num_images_per_prompt = num_images_per_prompt
        self._output_dir = output_dir
        self.torch_dtype = None
        self.accelerator = None
        self.device = None
        self.refiner = None
        self.pipe = None
        self.setup_pipeline()

    @property
    def use_refiner(self):
        return self._use_refiner

    @use_refiner.setter
    def use_refiner(self, value: bool):
        self._use_refiner = value
        self.setup_pipeline()

    @property
    def height(self):
        return self._height

    @height.setter
    def height(self, value: int):
        self._height = value

    @property
    def width(self):
        return self._width

    @width.setter
    def width(self, value: int):
        self._width = value

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @guidance_scale.setter
    def guidance_scale(self, value: float):
        self._guidance_scale = value

    @property
    def num_images_per_prompt(self):
        return self._num_images_per_prompt

    @num_images_per_prompt.setter
    def num_images_per_prompt(self, value: int):
        self._num_images_per_prompt = value

    @property
    def output_dir(self):
        return self._output_dir

    @output_dir.setter
    def output_dir(self, value: str):
        self._output_dir = value

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

        if self._use_refiner:
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
            self.refiner = None

    def process_prompts(self, prompts: List[Union[str, Dict]]):
        for prompt in prompts:
            if isinstance(prompt, str):
                self.generate_image(prompt)
            elif isinstance(prompt, dict):
                self.generate_image(**prompt)

    def generate_image(self, prompt: str, prompt_2: Optional[str] = None,
                       negative_prompt: Optional[str] = None, negative_prompt_2: Optional[str] = None,
                       seed: Optional[int] = None):
        if seed is None:
            seed = random.randint(0, sys.maxsize)

        generator = torch.Generator(self.device).manual_seed(seed)

        print(f"Prompt:\t{prompt}")
        print(f"Seed:\t{seed}")

        output_type = "latent" if self._use_refiner else "pil"
        images = self.pipe(
            prompt=prompt,
            prompt_2=prompt_2,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt_2,
            output_type=output_type,
            generator=generator,
            height=self._height,
            width=self._width,
            guidance_scale=self._guidance_scale,
            num_images_per_prompt=self._num_images_per_prompt
        ).images

        if self._use_refiner:
            images = self.refiner(prompt=prompt, negative_prompt=negative_prompt, image=images).images

        self._save_images(images, prompt)
        return images

    def _save_images(self, images, prompt):
        os.makedirs(self._output_dir, exist_ok=True)
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

        for i, image in enumerate(images):
            image_path = os.path.join(self._output_dir, f"output_{timestamp}_{i}.jpg")
            print(f"Saving image to {image_path}")
            image.save(image_path)

        print(f"Image complete. {prompt}")


# Usage example
if __name__ == "__main__":
    pipeline = StableDiffusionXLPipeline(use_refiner=False,
                                         num_images_per_prompt=1,
                                         output_dir="my_images")

    # Generate a single image using a string prompt
    pipeline.generate_image("A fairy princess and her majestic dragon. Photorealistic.")

    # Change some properties
    pipeline.use_refiner = True

    # Generate an image with more parameters
    pipeline.generate_image(
        prompt="A cyberpunk cityscape at night",
        negative_prompt="daytime, bright, sunny",
        seed=42
    )

    # Process multiple prompts
    prompts_to_process = [
        "A serene lake surrounded by mountains",
        {
            "prompt": "An alien landscape with two moons",
            "prompt_2": "Highly detailed, science fiction art",
            "negative_prompt": "Earth-like, familiar"
        }
    ]

    pipeline.process_prompts(prompts_to_process)
