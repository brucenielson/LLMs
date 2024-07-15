from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
import torch
from huggingface_hub import login

# Use huggingface-cli login
secret_file = r'D:\Documents\Secrets\huggingface_secret.txt'
try:
    with open(secret_file, 'r') as file:
        secret_text = file.read()
except FileNotFoundError:
    print(f"The file '{secret_file}' does not exist.")
except Exception as e:
    print(f"An error occurred: {e}")

login(secret_text)

tokenizer = AutoTokenizer.from_pretrained("google/gemma-1.1-2b-it")  # google/gemma-2-9b-it
model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-1.1-2b-it",
    device_map="auto",
    torch_dtype=torch.bfloat16
)

input_text = "Write me a poem about Machine Learning."
inputs = tokenizer(input_text, return_tensors="pt").to("cuda")
# https://huggingface.co/docs/transformers/v4.42.0/en/internal/generation_utils#transformers.TextStreamer
# https://huggingface.co/docs/text-generation-inference/conceptual/streaming
streamer = TextStreamer(tokenizer, skip_prompt=True, max_length=2000)

_ = model.generate(**inputs, streamer=streamer, max_length=2000, do_sample=True, temperature=0.9)

