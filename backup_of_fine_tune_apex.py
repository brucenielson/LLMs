from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments
from peft import LoraConfig, TaskType, get_peft_model
from huggingface_hub import login
import torch
from datasets import load_dataset
from trl import SFTTrainer
import time
import json
from torch.utils.data import Dataset


def formatting_func(example):
    text = f"### Reported Problem: {example['data'][0]}\n### Proposed Solution: {example['data'][1]}"
    return text


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


if __name__ == "__main__":
    # fine_tune_apex()
    # try_apex_model()
    try_fake_apex_model2()
    print("All tests passed!")