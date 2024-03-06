from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments
from peft import LoraConfig, TaskType, get_peft_model
from huggingface_hub import login
import torch
from datasets import load_dataset
from trl import SFTTrainer
import time
import json

from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self, data, tokenizer, max_seq_length):
        self.data = data
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # Tokenize text data
        inputs = self.tokenizer(item['data'], max_length=self.max_seq_length, truncation=True, padding='max_length', return_tensors='pt')

        # Add labels if available
        labels = None
        if 'labels' in item:
            labels = torch.tensor(item['labels'])

        return {
            'input_ids': inputs['input_ids'].squeeze(),
            'attention_mask': inputs['attention_mask'].squeeze(),
            'labels': labels,
        }


# pip installs
# Installing Pytorch
# https://pytorch.org/get-started/locally/
# conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
# Install Huggingface
# https://huggingface.co/docs/transformers/installation
# pip install transformers vs pip install 'transformers[torch]' vs conda install conda-forge::transformers
# Install hugging face hub
# python -m pip install huggingface_hub
# Install peft
# https://huggingface.co/docs/peft/install
# pip install peft
# Cuda installation for windows
# https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html
# conda install cuda -c nvidia
# try_peft() works so far
# Install bitsandbytes
# transformers.is_bitsandbytes_available() = False (at this point)
# torch.cuda.is_available() = True (at this point)
# Tried:
# Attempt 1
# https://github.com/jllllll/bitsandbytes-windows-webui
# python -m pip install bitsandbytes --prefer-binary --extra-index-url=https://jllllll.github.io/bitsandbytes-windows-webui
# transformers.is_bitsandbytes_available() = True
# But try_peft() now fails!
# Fails on model = get_peft_model(model, lora_config)
# pip uninstall bitsandbytes fixes the problem and try_peft() works again
# Attempt 2
# https://anaconda.org/conda-forge/bitsandbytes
# conda install conda-forge::bitsandbytes
# PackagesNotFoundError: The following packages are not available from current channels:
# transformers.is_bitsandbytes_available() = False / try_peft() still works - so basically nothing installed
# Attempt 3
# https://github.com/TimDettmers/bitsandbytes/issues/822
# python.exe -m pip install https://github.com/jllllll/bitsandbytes-windows-webui/releases/download/wheels/bitsandbytes-0.41.1-py3-none-win_amd64.whl
# transformers.is_bitsandbytes_available() = True
# try_peft() passes!!!! This is our best setup yet. I'm going to make a backup
# try_peft2() passes!! However, I get this ominous sounding warning:
# You are calling `save_pretrained` to a 4-bit converted model, but your `bitsandbytes` version doesn't support it. If you want to save 4-bit models, make sure to have `bitsandbytes>=0.41.3` installed.
# pip install trl



# Did not try:
# https://github.com/TimDettmers/bitsandbytes/issues/128
# https://www.reddit.com/r/LocalLLaMA/comments/13v57sf/bitsandbytes_giving_you_a_cuda_error_on_windows/
# https://pypi.org/project/bitsandbytes-windows/
# https://github.com/Keith-Hon/bitsandbytes-windows?tab=readme-ov-file
# https://github.com/Keith-Hon/bitsandbytes-windows


# Cmake your own version for windows
# https://huggingface.co/docs/bitsandbytes/main/en/installation?OS+system=Windows
# Requires Cmake that we can't find even though it is loaded
# https://cmake.org/download/
# https://stackoverflow.com/questions/70178963/where-is-cmake-located-when-downloaded-from-visual-studio-2022


# Less Important
# https://stackoverflow.com/questions/76924239/accelerate-and-bitsandbytes-is-needed-to-install-but-i-did
# # https://medium.com/@leennewlife/how-to-setup-pytorch-with-cuda-in-windows-11-635dfa56724b

# LORA Example
# https://colab.research.google.com/github/huggingface/peft/blob/main/examples/image_classification/image_classification_peft_lora.ipynb#scrollTo=3J5DokIqi-wV
# https://colab.research.google.com/drive/1vIjBtePIZwUaHWfjfNHzBjwuXOyU_ugD?usp=sharing#scrollTo=R4doTpXEyuVJ
# https://huggingface.co/docs/peft/v0.8.2/en/task_guides/image_classification_lora


# Cuda
# https://developer.nvidia.com/cuda-12-1-0-download-archive?target_os=Windows&target_arch=x86_64&target_version=11&target_type=exe_local
# https://github.com/NVIDIA/cudnn-frontend
# https://developer.nvidia.com/cudnn-downloads
# https://developer.nvidia.com/cudnn
# https://anaconda.org/nvidia/repo/installers?access=public&label=cuda-12.3.2&type=conda
# https://anaconda.org/nvidia/repo/installers?label=cuda-12.3.2&type=pypi
# https://www.nvidia.com/download/index.aspx?lang=en-us (Driver download)
# https://developer.nvidia.com/cuda-downloads

# Conda
# https://conda.io/projects/conda/en/latest/user-guide/getting-started.html

def formatting_func(example):
    text = f"### USER: {example['data'][0]}\n### ASSISTANT: {example['data'][1]}"
    return text


def try_peft():
    # From https://pytorch.org/blog/finetune-llms/
    # Model https://huggingface.co/meta-llama/Llama-2-7b-hf
    # Model https://llama.meta.com/llama-downloads/
    # Token found here to login https://huggingface.co/settings/tokens
    # Documentation https://huggingface.co/docs/transformers/main/model_doc/llama2
    # https://huggingface.co/meta-llama/Llama-2-7b-hf

    # Use huggingface-cli login
    # secret_file = r'D:\Projects\Holley\huggingface_secret.txt'
    # try:
    #     with open(secret_file, 'r') as file:
    #         secret_text = file.read()
    #         print(secret_text)
    # except FileNotFoundError:
    #     print(f"The file '{secret_file}' does not exist.")
    # except Exception as e:
    #     print(f"An error occurred: {e}")
    #
    # login(secret_text)
    # Base Model
    model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")

    # Create peft config
    lora_config = LoraConfig(
        r=8,
        target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_prog", "down_proj"],
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    # Create PeftModel which inserts LoRA adapters using the above config
    model = get_peft_model(model, lora_config)

    # Train the _model using HF Trainer / HF Accelerate/ cusom loop

    # Save the adapter weights
    # _model.save_pretrained("my_awesome_adapter")
    pass


    # From https://colab.research.google.com/drive/1vIjBtePIZwUaHWfjfNHzBjwuXOyU_ugD?usp=sharing#scrollTo=-6_Sx4NhI35N
    # See https://pytorch.org/blog/finetune-llms/
    # https://github.com/facebookresearch/llama?fbclid=IwAR3k8LPltMZmsRUnId-Oi5ZzLZf7JjdZjvsT5OyK0rV5y15HHW92ZSEhlXg
    # Read huggingface secret from this file: D:\Documents\Papers\EPub Books\huggingface_secret.txt
    # secret_file = r'D:\Projects\Holley\huggingface_secret.txt'
    # try:
    #     with open(secret_file, 'r') as file:
    #         secret_text = file.read()
    #         print(secret_text)
    # except FileNotFoundError:
    #     print(f"The file '{secret_file}' does not exist.")
    # except Exception as e:
    #     print(f"An error occurred: {e}")
    #
    # login(secret_text)
    # Load the 7b llama _model
def try_peft2():
    model_id = "meta-llama/Llama-2-7b-hf"

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=False,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
    )

    # Load _model - This line fails because it can't find CUDA
    model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=quantization_config)
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    # Set it to a new token to correctly attend to EOS tokens.
    tokenizer.add_special_tokens({'pad_token': '<PAD>'})
    lora_config = LoraConfig(
        r=8,
        target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )

    model.add_adapter(lora_config)
    train_dataset = load_dataset("stingning/ultrachat", split="train[:1%]")

    output_dir = r"D:\Projects\Holley\Apex AI Agent\apexCopilot\llama-7b-qlora-ultrachat-tutorial2"
    per_device_train_batch_size = 4
    gradient_accumulation_steps = 4
    optim = "paged_adamw_32bit"
    save_steps = 10
    logging_steps = 10
    learning_rate = 2e-4
    max_grad_norm = 0.3
    max_steps = 20
    warmup_ratio = 0.03
    lr_scheduler_type = "constant"

    training_arguments = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        optim=optim,
        save_steps=save_steps,
        logging_steps=logging_steps,
        learning_rate=learning_rate,
        max_grad_norm=max_grad_norm,
        max_steps=max_steps,
        warmup_ratio=warmup_ratio,
        lr_scheduler_type=lr_scheduler_type,
        gradient_checkpointing=True,
        push_to_hub=False,
    )
    # Fails here
    trainer = SFTTrainer(
        model=model,
        args=training_arguments,
        train_dataset=train_dataset,
        packing=True,
        dataset_text_field="id",
        tokenizer=tokenizer,
        max_seq_length=1024,
        formatting_func=formatting_func,
    )
    trainer.train()
    pass


def try_peft3():
    if torch.cuda.is_available():
        # Set the device to GPU
        device = torch.device("cuda")
        print("GPU is available.")
    else:
        # Set the device to CPU if GPU is not available
        device = torch.device("cpu")
        print("GPU is not available. Using CPU.")

    model_id = "meta-llama/Llama-2-7b-hf"

    print("torch.cuda.memory_summary():")
    print(torch.cuda.memory_summary())

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=False,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
    )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    # Set it to a new token to correctly attend to EOS tokens.
    tokenizer.add_special_tokens({'pad_token': '<PAD>'})

    # Load _model - This line fails because it can't find CUDA
    model = AutoModelForCausalLM.from_pretrained(model_id) #, quantization_config=quantization_config)
    model.to(device)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    # Set it to a new token to correctly attend to EOS tokens.
    tokenizer.add_special_tokens({'pad_token': '<PAD>'})
    lora_config = LoraConfig(
        r=8,
        target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )

    model.add_adapter(lora_config)
    train_dataset = load_dataset("stingning/ultrachat", split="train[:1%]")

    output_dir = r"D:\Projects\Holley\Apex AI Agent\apexCopilot\llama-7b-qlora-ultrachat-tutorial2"
    per_device_train_batch_size = 2
    gradient_accumulation_steps = 10
    optim = "paged_adamw_32bit"
    save_steps = 10
    logging_steps = 10
    learning_rate = 2e-4
    max_grad_norm = 0.3
    max_steps = 20
    warmup_ratio = 0.03
    lr_scheduler_type = "constant"

    training_arguments = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        optim=optim,
        save_steps=save_steps,
        logging_steps=logging_steps,
        learning_rate=learning_rate,
        max_grad_norm=max_grad_norm,
        max_steps=max_steps,
        warmup_ratio=warmup_ratio,
        lr_scheduler_type=lr_scheduler_type,
        gradient_checkpointing=False,
        push_to_hub=False,
    )
    # Fails here
    trainer = SFTTrainer(
        model=model,
        args=training_arguments,
        train_dataset=train_dataset,
        packing=True,
        dataset_text_field="id",
        tokenizer=tokenizer,
        max_seq_length=1024,
        formatting_func=formatting_func,
    )
    trainer.train()
    pass


def try_model():
    model_id = "ybelkada/llama-7b-qlora-ultrachat"
    # model_id = r"D:\Projects\Holley\Apex AI Agent\apexCopilot\llama-7b-qlora-ultrachat-tutorial1\checkpoint-1000"

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    quantization_config = BitsAndBytesConfig(load_in_4bit=False, bnb_4bit_compute_dtype=torch.float16)

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=quantization_config,
        adapter_kwargs={"revision": "09487e6ffdcc75838b10b6138b6149c36183164e"}
    )

    text = "### USER: Can you explain contrastive learning in machine learning in simple terms for someone new to the field of ML?### Assistant:"

    inputs = tokenizer(text, return_tensors="pt").to(0)
    outputs = model.generate(inputs.input_ids, max_new_tokens=250, do_sample=False)

    print("After attaching Lora adapters:")
    print(tokenizer.decode(outputs[0], skip_special_tokens=False))

    model.disable_adapters()
    outputs = model.generate(inputs.input_ids, max_new_tokens=250, do_sample=False)

    print("Before Lora:")
    print(tokenizer.decode(outputs[0], skip_special_tokens=False))


def try_llama():
    model_id = "meta-llama/Llama-2-7b-hf"
    model = AutoModelForCausalLM.from_pretrained(model_id).to("cuda")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    prompt = "Once upon a time, there was a brave knight who"
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to("cuda")
    # Generate text
    output = model.generate(input_ids, max_length=200, num_return_sequences=1)
    # Decode the generated tokens back to text
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    print(generated_text)


def try_llama_cuda():
    model_id = "meta-llama/Llama-2-7b-hf"
    model = AutoModelForCausalLM.from_pretrained(model_id).to("cuda")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    prompt = "Once upon a time, there was a brave knight who"
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to("cuda")

    start_time = time.time()
    output = model.generate(input_ids, max_length=1000, num_return_sequences=1)
    end_time = time.time()

    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    print("CUDA execution time (ms):", (end_time - start_time) * 1000)
    print(generated_text)


def try_llama_cpu():
    model_id = "meta-llama/Llama-2-7b-hf"
    model = AutoModelForCausalLM.from_pretrained(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    prompt = "Once upon a time, there was a brave knight who"
    input_ids = tokenizer.encode(prompt, return_tensors="pt")

    start_time = time.time()
    output = model.generate(input_ids, max_length=1000, num_return_sequences=1)
    end_time = time.time()

    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    print("Non-CUDA execution time (ms):", (end_time - start_time) * 1000)
    print(generated_text)


def load_apex_data():
    # Load D:\Projects\Holley\Apex Copilot\bin\apex_data.json
    file_path = r'D:\Projects\Holley\Apex Copilot\bin\apex_data.json'
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data_list = json.load(file)
    except UnicodeDecodeError as e:
        print(f"UnicodeDecodeError: {e}")
        print(e)

    # Now, data_list contains the contents of the JSON file as a list
    return data_list


def fine_tune_apex():
    data = load_apex_data()
    model_id = "meta-llama/Llama-2-7b-hf"

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
    )

    # Load model
    model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=quantization_config)
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    # Set it to a new token to correctly attend to EOS tokens.
    tokenizer.add_special_tokens({'pad_token': '<PAD>'})
    lora_config = LoraConfig(
        r=8,
        target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )

    model.add_adapter(lora_config)
    train_dataset = [{'id': str(item['CaseId']), 'data': ['The customer reports this problem: ' + item['Problem'],
                                                          'Suggested Solution: ' + item['Resolution']]} for item in data]

    # train_dataset = train_dataset[0:100]

    output_dir = r"D:\Projects\Holley\Apex AI Agent\apexCopilot\apex_copilot_model"
    per_device_train_batch_size = 4
    gradient_accumulation_steps = 4
    optim = "paged_adamw_32bit"
    save_steps = 10
    logging_steps = 10
    learning_rate = 2e-5
    max_grad_norm = 0.3
    max_steps = 10
    warmup_ratio = 0.03
    lr_scheduler_type = "constant"

    training_arguments = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        optim=optim,
        save_steps=save_steps,
        logging_steps=logging_steps,
        learning_rate=learning_rate,
        max_grad_norm=max_grad_norm,
        max_steps=max_steps,
        warmup_ratio=warmup_ratio,
        lr_scheduler_type=lr_scheduler_type,
        gradient_checkpointing=True,
        push_to_hub=False,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_arguments,
        train_dataset=train_dataset,
        packing=True,
        dataset_text_field="id",
        tokenizer=tokenizer,
        max_seq_length=1024,
        formatting_func=formatting_func,
    )
    trainer.train()

def try_apex_model():
    model_id = r"D:\Projects\Holley\Apex AI Agent\apexCopilot\apex_copilot_model\checkpoint-360"

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4"
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=quantization_config,
        adapter_kwargs={"revision": "09487e6ffdcc75838b10b6138b6149c36183164e"}
    )

    text = ("You are a tech support agent. The customer will report a problem and you'll come back with suggested solutions."
            "The customer reports this problem: Customer is trying to program the truck for the first time. He is getting an Error code 1006. Customer has never programmed before.")

    inputs = tokenizer(text, return_tensors="pt").to(0)
    outputs = model.generate(inputs.input_ids, max_new_tokens=512, do_sample=True)

    print("After attaching Lora adapters:")
    print(tokenizer.decode(outputs[0], skip_special_tokens=False))

    model.disable_adapters()
    outputs = model.generate(inputs.input_ids, max_new_tokens=512, do_sample=True)

    print("Before Lora:")
    print(tokenizer.decode(outputs[0], skip_special_tokens=False))


def try_fake_apex_model():

    model_id = "meta-llama/Llama-2-7b-hf"

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4"
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        # quantization_config=quantization_config,
        # adapter_kwargs={"revision": "09487e6ffdcc75838b10b6138b6149c36183164e"}
    )

    text = ("You are a tech support agent. The customer will report a problem and you'll come back with suggested solutions."
            "The customer reports this problem: Customer is trying to program the truck for the first time. He is getting an Error code 1006. Customer has never programmed before.")

    inputs = tokenizer(text, return_tensors="pt")
    outputs = model.generate(inputs.input_ids, max_new_tokens=256, do_sample=True)

    print("After attaching Lora adapters:")
    print(tokenizer.decode(outputs[0], skip_special_tokens=False))


def try_fake_apex_model2():
    # model_id = "meta-llama/Llama-2-7b-hf"
    model_id = r"D:\Projects\Holley\Apex AI Agent\apexCopilot\apex_copilot_model\checkpoint-360"
    model = AutoModelForCausalLM.from_pretrained(model_id).to("cuda")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    prompt = ("You are a tech support agent. The customer will report a problem and you'll come back with suggested "
              "solutions. You will be given a prompt that starts with 'The customer resports this problem: ' followed "
              "by a description of the reported problem and any details available. You will respond with a suggested "
              "solution to the problem after the words 'Suggested Solution: ' "
              "Word the 'Suggested Solution' like you are a support agent talking to a "
              "customer in full English sentences. Give one or more suggested solutions to the customer."              
              "\nThe customer reports this problem: Customer is trying to program the truck for the "
              "first time. He is getting an Error code 1006. Customer has never programmed before. "
              "\nSuggested Solution: ")

    input_ids = tokenizer.encode(prompt, return_tensors="pt").to("cuda")

    start_time = time.time()
    output = model.generate(input_ids,
                            max_length=256,
                            num_return_sequences=2,
                            num_beams=3,
                            repetition_penalty=1.2
                            )
    end_time = time.time()

    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    print("CUDA execution time (ms):", (end_time - start_time) * 1000)
    print(generated_text)

"""Examples:
  {
    "CaseId": 11149,
    "CreationDateTime": "2008-12-17T16:14:05.793",
    "Problem": "Issue Category: Product\r\nIssue Sub Category: Error code\r\nDiagnosis Category: Vehicle\r\nDiagnosis: Vehicle hardware failure\r\nPart #: 15000\r\nProduct: SKU,Ford,Evolution Prg (gameboy),7.3L,94-03\r\nReported Problem: Customer is trying to program the truck for the first time. He is getting an Error code 1006. Customer has never programmed before.",
    "Resolution": "Resolution Category: Vehicle\r\nResolution Diagnosis Description: Vehicle hardware failure\r\nResolution Notes: See action taken.\r\nTold customer that the Error code says that the PCM is unresponsive, so more than likely he'll need to have the truck looked at or see if the PCM can be updated. Or check if there are any chips in the PCM."
  },
    {
    "CaseId": 11152,
    "CreationDateTime": "2008-12-17T16:56:55.71",
    "Problem": "Issue Category: Product\r\nIssue Sub Category: Missing/incorrect parts\r\nPart #: 25060\r\nProduct: SKU,Chevy,C/K Series Gas,Evolution 2.5,07-08\r\nReported Problem: Pod does not match dash",
    "Resolution": "Resolution Notes: making a Shipping request for correct pod\r\ncustomer needs a 28302 to fit truck"
  },
    {
    "CaseId": 11158,
    "CreationDateTime": "2008-12-18T07:58:38.77",
    "Problem": "Issue Category: Product\r\nIssue Sub Category: Error code\r\nDiagnosis Category: Product\r\nDiagnosis: Programming failed\r\nPart #: 15002\r\nProduct: Ford 6.0L 03-04 Evolution\r\nReported Problem: Customer gets an error code when trying to do custom tunes on his truck.",
    "Resolution": "Resolution Category: Product\r\nResolution Diagnosis Description: Programming failed\r\nResolution Notes: Customer was able to successfully program the vehicle.\r\nTargeted customer to a different firmware and calibration."
  },
  """



if __name__ == "__main__":
    # try_peft()
    # try_peft2()
    # fine_tune_apex()
    # try_model()
    # try_apex_model()
    try_fake_apex_model2()
    # try_llama()
    # try_llama_cuda()
    # try_llama_cpu()
    print("All tests passed!")


# Other links
# https://colab.research.google.com/github/brevdev/notebooks/blob/main/llama2-finetune-own-data.ipynb
# How to use peft
# https://github.com/huggingface/peft
# llama cpp python
# https://llama-cpp-python.readthedocs.io/en/latest/
# https://github.com/abetlen/llama-cpp-python
# Hugging face tutorial: https://dev.to/pavanbelagatti/hugging-face-101-a-tutorial-for-absolute-beginners-3b0l
# Cuda toolkit: https://developer.nvidia.com/cuda-downloads?target_os=Windows&target_arch=x86_64&target_version=11&target_type=exe_local
# Hugging face tokens: https://huggingface.co/settings/tokens
# https://python.langchain.com/docs/guides/local_llms
# https://pypi.org/project/transformers/
# https://semaphoreci.com/blog/local-llm
# https://colab.research.google.com/github/brevdev/notebooks/blob/main/llama2-finetune-own-data.ipynb
# https://llama.meta.com/get-started/
# https://sych.io/blog/how-to-run-llama-2-locally-a-guide-to-running-your-own-chatgpt-like-large-language-model/
# https://pypi.org/project/python-llm/
# https://www.signitysolutions.com/tech-insights/training-transformer-models-with-hugging-face

# Original list of links:
# https://pytorch.org/blog/finetune-llms/
# https://blog.maartenballiauw.be/post/2023/06/15/running-large-language-models-locally-your-own-chatgpt-like-ai-in-csharp.html
# https://github.blog/2023-10-30-the-architecture-of-todays-llm-applications/
# https://deci.ai/blog/small-giants-top-10-under-13b-llms-in-open-source/
# https://simonwillison.net/2023/Nov/29/llamafile/
# https://www.linkedin.com/pulse/how-llm-trained-mastering-llm-large-language-model
# https://venturebeat.com/ai/microsoft-releases-phi-2-a-small-language-model-ai-that-outperforms-llama-2-mistral-7b/
# https://bdtechtalks.com/2023/11/27/streamingllm/


# Fixing Conda enviornments
# https://stackoverflow.com/questions/57527131/conda-environment-has-no-name-visible-in-conda-env-list-how-do-i-activate-it-a1
# https://stackoverflow.com/questions/48924787/pycharm-terminal-doesnt-activate-conda-environment
# conda info --envs
# conda env remove
# https://stackoverflow.com/questions/49127834/removing-conda-environment
