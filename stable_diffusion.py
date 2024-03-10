import random
import sys
import torch
import os
import datetime
from diffusers import DiffusionPipeline
from accelerate import Accelerator


def setup_pipeline(prompts):
    # noinspection SpellCheckingInspection
    # https://medium.com/@demstalfer/quick-guide-to-using-the-new-stable-diffusion-xl-for-code-enthusiasts-bd71136dd794
    # https://huggingface.co/stabilityai/stable-diffusion-2-1
    use_refiner = True
    common_params = {
        "torch_dtype": torch.float16,
        "use_safetensors": True,
        "variant": "fp16",
        # "device_map": "auto",
    }

    accelerator = Accelerator()
    pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", **common_params)
    device = accelerator.device  # "cuda" if torch.cuda.is_available() else "cpu"
    pipe = pipe.to(device)

    refiner = None
    if use_refiner:
        refiner_params = {
            "text_encoder_2": pipe.text_encoder_2,
            "vae": pipe.vae,
            **common_params,
        }
        refiner = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-refiner-1.0", **refiner_params)
        refiner = refiner.to(device)
        # pipe.enable_model_cpu_offload()
    else:
        pipe = pipe.to(device)

    for prompt in prompts:
        seed = random.randint(0, sys.maxsize)

        print(f"Prompt:\t{prompt}")
        print(f"Seed:\t{seed}")

        pipeline_params = {
            "prompt": prompt,
            "output_type": "latent" if use_refiner else "pil",
            "generator": torch.Generator(device).manual_seed(seed),
        }

        pipe = pipe.to(device)
        images = pipe(**pipeline_params).images

        if use_refiner:
            refiner_params = {
                "prompt": prompt,
                "image": images,
            }
            images = refiner(**refiner_params).images

        output_dir = 'output_images'
        os.makedirs(output_dir, exist_ok=True)

        # Save all images
        for i, image in enumerate(images):
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            image_path = os.path.join(output_dir, f"output_{timestamp}_{i}.jpg")
            print(f"Saving image to {image_path}")
            image = image.to("cpu")
            image.save(image_path)

        print(f"Image complete. {prompt}")


# List of prompts to be processed
prompts_to_process = [
    'Superman saves the day',

    # Add more prompts as needed
]

# Run continuously
while True:
    setup_pipeline(prompts_to_process)
