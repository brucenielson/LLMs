import random
import sys
import torch
import os
import datetime
from diffusers import DiffusionPipeline
from accelerate import Accelerator


def setup_pipeline(prompts, use_refiner=True):
    torch_dtype = torch.float16
    use_safetensors = True
    variant = "fp16"

    accelerator = Accelerator()
    device = accelerator.device

    pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0",
                                             device=device,
                                             torch_dtype=torch_dtype,
                                             use_safetensors=use_safetensors,
                                             variant=variant)

    refiner = None
    if use_refiner:
        refiner = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-refiner-1.0",
                                                    text_encoder_2=pipe.text_encoder_2,
                                                    vae=pipe.vae,
                                                    device=device,
                                                    torch_dtype=torch_dtype,
                                                    use_safetensors=use_safetensors,
                                                    variant=variant).to(device)
        pipe.enable_model_cpu_offload()

    for prompt in prompts:
        seed = random.randint(0, sys.maxsize)

        print(f"Prompt:\t{prompt}")
        print(f"Seed:\t{seed}")

        generator = torch.Generator(device).manual_seed(seed)

        output_type = "latent" if use_refiner else "pil"

        images = pipe(prompt=prompt, output_type=output_type, generator=generator).images

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
    "Superman to the rescue."
]

setup_pipeline(prompts_to_process)
