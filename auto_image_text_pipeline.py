from diffusers import AutoPipelineForText2Image
import torch

pipeline = AutoPipelineForText2Image.from_pretrained(
    "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16, use_safetensors=True
).to("cuda")
prompt = "Superman to the rescue!"

image = pipeline(prompt, num_inference_steps=25).images[0]
# save image off to output_images folder
image.save("output_images/output.jpg")
