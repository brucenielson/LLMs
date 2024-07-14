from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from huggingface_hub import login

# Use huggingface-cli login
secret_file = r'D:\Documents\Secrets\huggingface_secret.txt'
try:
    with open(secret_file, 'r') as file:
        secret_text = file.read()
        print(secret_text)
except FileNotFoundError:
    print(f"The file '{secret_file}' does not exist.")
except Exception as e:
    print(f"An error occurred: {e}")

login(secret_text)

tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-9b-it")  # google/gemma-1.1-2b-it
model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-2-9b-it",
    device_map="auto",
    torch_dtype=torch.bfloat16
)

input_text = "Write me a poem about Machine Learning."
input_ids = tokenizer(input_text, return_tensors="pt").to("cuda")

outputs = model.generate(**input_ids)
print(tokenizer.decode(outputs[0]))