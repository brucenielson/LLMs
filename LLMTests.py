from transformers import pipeline
import requests
from PIL import Image
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, TaskType, get_peft_model
from llama_cpp import Llama
from huggingface_hub import notebook_login, login
import torch


def test_sentiment_analysis():
    # From https://pypi.org/project/transformers/
    classifier = pipeline("sentiment-analysis")
    result = classifier('We are very happy to introduce pipeline to the transformers repository.')
    assert result[0]['label'] == 'POSITIVE'
    assert result[0]['score'] > 0.999

    result = classifier("I hate you")[0]
    assert result['label'] == 'NEGATIVE'
    assert result['score'] > 0.5

    result = classifier("I love you")[0]
    assert result['label'] == 'POSITIVE'


def test_computer_vision():
    # From https://pypi.org/project/transformers/

    # Download an image with cute cats
    url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/coco_sample.png"
    image_data = requests.get(url, stream=True).raw
    image = Image.open(image_data)

    # Allocate a pipeline for object detection
    object_detector = pipeline('object-detection')
    result = object_detector(image)
    # Assert detector found the cats
    # [{'score': 0.9982201457023621,
    #   'label': 'remote',
    #   'box': {'xmin': 40, 'ymin': 70, 'xmax': 175, 'ymax': 117}},
    #  {'score': 0.9960021376609802,
    #   'label': 'remote',
    #   'box': {'xmin': 333, 'ymin': 72, 'xmax': 368, 'ymax': 187}},
    #  {'score': 0.9954745173454285,
    #   'label': 'couch',
    #   'box': {'xmin': 0, 'ymin': 1, 'xmax': 639, 'ymax': 473}},
    #  {'score': 0.9988006353378296,
    #   'label': 'cat',
    #   'box': {'xmin': 13, 'ymin': 52, 'xmax': 314, 'ymax': 470}},
    #  {'score': 0.9986783862113953,
    #   'label': 'cat',
    #   'box': {'xmin': 345, 'ymin': 23, 'xmax': 640, 'ymax': 368}}]
    assert len(result) == 5
    assert result[3]['label'] == 'cat'
    assert result[4]['label'] == 'cat'


def test_download_model():
    # https://pypi.org/project/transformers/
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModel.from_pretrained("bert-base-uncased")
    print_trainable_parameters(model)
    inputs = tokenizer("Hello world!", return_tensors="pt")
    outputs = model(**inputs)
    print(outputs)


def test_llama_cpp():
    # From https://github.com/abetlen/llama-cpp-python
    model_path = r'E:\LLMs\TheBloke\Mistral-7B-Instruct-v0.1-GGUF\mistral-7b-instruct-v0.1.Q4_0.gguf'
    # model_path = r"E:\LLMs\TheBloke\Chronomaid-Storytelling-13B-GGUF\chronomaid-storytelling-13b.Q5_K_M.gguf"
    # model_path = r"E:\LLMs\TheBloke\openchat_3.5-GGUF\openchat_3.5.Q8_0.gguf"
    llm = Llama(
        model_path=model_path,
        n_gpu_layers=6,  # Uncomment to use GPU acceleration
        # seed=1337, # Uncomment to set a specific seed
        # n_ctx=2048, # Uncomment to increase the context window
    )
    print("Model loaded!")
    output = llm(
          "Q: Name the planets in the solar system? A: ",  # Prompt
          max_tokens=32,  # Generate up to 32 tokens, set to None to generate up to the end of the context window
          stop=["Q:", "\n"],  # Stop generating just before the _model would generate a new query
          echo=True  # Echo the prompt back in the output
    )
    print(output)


def test_llama_cpp_chat_completion():
    model_path = r'E:\LLMs\TheBloke\Mistral-7B-Instruct-v0.1-GGUF\mistral-7b-instruct-v0.1.Q4_0.gguf'
    llm = Llama(
        model_path=model_path,
        n_gpu_layers=6,
        chat_format="llama-2"
    )
    print("Model loaded!")
    result = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": "You are a story teller."},
            {
                "role": "user",
                "content": "Write a fantasy story in the style of Robert E. Howard's Conan the Barbarian."
            }
        ],
        max_tokens=8192
    )
    print(result['choices'][0]['message']['content'])


def test_llama_llm():
    model_path = r'E:\LLMs\TheBloke\Mistral-7B-Instruct-v0.1-GGUF\mistral-7b-instruct-v0.1.Q4_0.gguf'
    llm = Llama(
        model_path=model_path,
        n_gpu_layers=4,  # Uncomment to use GPU acceleration
        # seed=1337, # Uncomment to set a specific seed
        # n_ctx=2048, # Uncomment to increase the context window
    )
    print("Model loaded!")
    output = llm(
          "Q: Name the planets in the solar system? A: ",
          max_tokens=4096,  # Generate up to 32 tokens, set to None to generate up to the end of the context window
    )
    # print(output)
    print(output['choices'][0]['text'])


def test_download_transformer():
    # try:
    #     pipe = pipeline("text-generation", _model="TheBloke/Chronomaid-Storytelling-13B-GPTQ")
    # except Exception as e:
    #     print("Model not found!")
    #
    # print("Model down loaded!")
    # result = pipe("Q: Name the planets in the solar system? A: ")
    # print(result)
    # _model = AutoModel.from_pretrained("TheBloke/Chronomaid-Storytelling-13B-GGUF")
    # print(_model)
    # pipe = pipeline("text-generation", _model="allenai/OLMo-7B")
    # generated_text = pipe("Once upon a time, there was a brave knight who")
    # print(generated_text)
    pass


def test_load_in_bit():
    model_id = "facebook/opt-125m"
    # For LLM.int8()
    # _model = AutoModelForCausalLM.from_pretrained(model_id, load_in_8bit=True)

    # For QLoRA
    model = AutoModelForCausalLM.from_pretrained(model_id, load_in_4bit=True)
    print(model)


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the _model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )


def test_llama_2():
    model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
    prompt = "Once upon a time, there was a brave knight who"
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    # Generate text
    output = model.generate(input_ids, max_length=1024, num_return_sequences=1)
    # Decode the generated tokens back to text
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    print(generated_text)


def test_gpt2():
    pipe = pipeline("text-generation", model="gpt2")
    query = "Once upon a time, there was a brave knight who"
    result = pipe(query, max_length=4096)
    print(result[0]['generated_text'])


def test_peft():
    # From https://pytorch.org/blog/finetune-llms/
    # Model https://huggingface.co/meta-llama/Llama-2-7b-hf
    # Model https://llama.meta.com/llama-downloads/
    # Token found here to login https://huggingface.co/settings/tokens
    # Documentation https://huggingface.co/docs/transformers/main/model_doc/llama2
    # https://huggingface.co/meta-llama/Llama-2-7b-hf

    # Use huggingface-cli login
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


def test_peft2():
    pass
    # From https://colab.research.google.com/drive/1vIjBtePIZwUaHWfjfNHzBjwuXOyU_ugD?usp=sharing#scrollTo=-6_Sx4NhI35N
    # See https://pytorch.org/blog/finetune-llms/
    # https://github.com/facebookresearch/llama?fbclid=IwAR3k8LPltMZmsRUnId-Oi5ZzLZf7JjdZjvsT5OyK0rV5y15HHW92ZSEhlXg
    # Read huggingface secret from this file: D:\Documents\Papers\EPub Books\huggingface_secret.txt
    secret_file = r'D:\Documents\Papers\EPub Books\huggingface_secret.txt'
    try:
        with open(secret_file, 'r') as file:
            secret_text = file.read()
            print(secret_text)
    except FileNotFoundError:
        print(f"The file '{secret_file}' does not exist.")
    except Exception as e:
        print(f"An error occurred: {e}")

    login(secret_text)
    # Load the 7b llama _model
    model_id = "meta-llama/Llama-2-7b-hf"

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4"
    )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    # Set it to a new token to correctly attend to EOS tokens.
    tokenizer.add_special_tokens({'pad_token': '<PAD>'})

    # Load _model
    model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=quantization_config)


if __name__ == "__main__":
    # test_sentiment_analysis()
    # test_computer_vision()
    # test_download_model()
    # test_llama_cpp()
    # test_llama_cpp_chat_completion()
    # test_llama_llm()
    # test_gpt2()
    # Failed
    # test_download_transformer()
    # test_load_in_bit()
    # test_llama_2()
    # test_peft()
    test_peft2()
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
