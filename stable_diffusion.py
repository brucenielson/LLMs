import mediapy as media
import random
import sys
import torch
import os
import datetime
from diffusers import DiffusionPipeline


def setup_pipline():
    # https://medium.com/@demstalfer/quick-guide-to-using-the-new-stable-diffusion-xl-for-code-enthusiasts-bd71136dd794
    use_refiner = False
    common_params = {
        "torch_dtype": torch.float16,
        "use_safetensors": True,
        "variant": "fp16",
    }


    pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", **common_params)


    device = "cuda"


    if use_refiner:
        refiner_params = {
            "text_encoder_2": pipe.text_encoder_2,
            "vae": pipe.vae,
            **common_params,
        }
        refiner = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-refiner-1.0", **refiner_params)


        refiner = refiner.to(device)
        pipe.enable_model_cpu_offload()
    else:
        pipe = pipe.to(device)

    prompt = "Elon Musk and Mark Zuckerberg fighting in a mma ring"
    seed = random.randint(0, sys.maxsize)


    pipeline_params = {
        "prompt": prompt,
        "output_type": "latent" if use_refiner else "pil",
        "generator": torch.Generator("cuda").manual_seed(seed),
    }


    images = pipe(**pipeline_params).images


    if use_refiner:
        refiner_params = {
            "prompt": prompt,
            "image": images,
        }
        images = refiner(**refiner_params).images


    print(f"Prompt:\t{prompt}")
    print(f"Seed:\t{seed}")


    output_dir = 'output_images'
    os.makedirs(output_dir, exist_ok=True)


    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    image_path = os.path.join(output_dir, f"output_{timestamp}.jpg")
    images[0].save(image_path)


    media.show_images(images)


    print(f"Image saved at: {image_path}")

setup_pipline()