import random
import sys
import torch
import os
import datetime
from diffusers import DiffusionPipeline
from accelerate import Accelerator


def setup_pipeline(prompts, use_refiner=True, height=None, width=None, guidance_scale=5.0, num_images_per_prompt=1):
    torch_dtype = torch.float16
    use_safetensors = True
    variant = "fp16"
    output_type = "latent" if use_refiner else "pil"

    accelerator = Accelerator()
    device = accelerator.device

    pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0",
                                             torch_dtype=torch_dtype,
                                             use_safetensors=use_safetensors,
                                             variant=variant)

    refiner = None
    if use_refiner:
        refiner = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-refiner-1.0",
                                                    text_encoder_2=pipe.text_encoder_2,
                                                    vae=pipe.vae,
                                                    torch_dtype=torch_dtype,
                                                    use_safetensors=use_safetensors,
                                                    variant=variant).to(device)
        pipe.enable_model_cpu_offload()
    else:
        pipe = pipe.to(device)

    for prompt_tuple in prompts:
        if isinstance(prompt_tuple, str):
            prompt = prompt_tuple
            prompt_2 = None
            negative_prompt = None
            negative_prompt_2 = None
        else:
            prompt = prompt_tuple[0]
            prompt_2 = prompt_tuple[1] if len(prompt_tuple) > 1 else None
            negative_prompt = prompt_tuple[2] if len(prompt_tuple) > 2 else None
            negative_prompt_2 = prompt_tuple[3] if len(prompt_tuple) > 3 else None

        seed = random.randint(0, sys.maxsize)

        print(f"Prompt:\t{prompt}")
        print(f"Seed:\t{seed}")

        generator = torch.Generator(device).manual_seed(seed)

        images = pipe(prompt=prompt, prompt_2=prompt_2, negative_prompt=negative_prompt,
                      negative_prompt_2=negative_prompt_2, output_type=output_type, generator=generator,
                      height=height, width=width, guidance_scale=guidance_scale,
                      num_images_per_prompt=num_images_per_prompt).images

        if use_refiner:
            images = refiner(prompt=prompt, image=images).images

        output_dir = 'output_images'
        os.makedirs(output_dir, exist_ok=True)

        for i, image in enumerate(images):
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            image_path = os.path.join(output_dir, f"output_{timestamp}_{i}.jpg")
            print(f"Saving image to {image_path}")
            image.save(image_path)

        print(f"Image complete. {prompt}")


prompts_to_process = [
    "A fairy princess and her majestic dragon. Photorealistic."
]

setup_pipeline(prompts_to_process) #, num_images_per_prompt=2)
# https://huggingface.co/stabilityai/stable-diffusion-xl-refiner-1.0
# https://huggingface.co/docs/diffusers/main/en/api/pipelines/stable_diffusion/stable_diffusion_xl