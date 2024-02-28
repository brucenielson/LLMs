from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, TaskType, get_peft_model
from huggingface_hub import login
import torch

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
# To Try:
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


def try_peft():
    # From https://pytorch.org/blog/finetune-llms/
    # Model https://huggingface.co/meta-llama/Llama-2-7b-hf
    # Model https://llama.meta.com/llama-downloads/
    # Token found here to login https://huggingface.co/settings/tokens
    # Documentation https://huggingface.co/docs/transformers/main/model_doc/llama2
    # https://huggingface.co/meta-llama/Llama-2-7b-hf

    # Use huggingface-cli login
    secret_file = r'D:\Projects\Holley\huggingface_secret.txt'
    try:
        with open(secret_file, 'r') as file:
            secret_text = file.read()
            print(secret_text)
    except FileNotFoundError:
        print(f"The file '{secret_file}' does not exist.")
    except Exception as e:
        print(f"An error occurred: {e}")

    login(secret_text)
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


def try_peft2():
    # From https://colab.research.google.com/drive/1vIjBtePIZwUaHWfjfNHzBjwuXOyU_ugD?usp=sharing#scrollTo=-6_Sx4NhI35N
    # See https://pytorch.org/blog/finetune-llms/
    # https://github.com/facebookresearch/llama?fbclid=IwAR3k8LPltMZmsRUnId-Oi5ZzLZf7JjdZjvsT5OyK0rV5y15HHW92ZSEhlXg
    # Read huggingface secret from this file: D:\Documents\Papers\EPub Books\huggingface_secret.txt
    secret_file = r'D:\Projects\Holley\huggingface_secret.txt'
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

    # Load _model - This line fails because it can't find CUDA
    model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=quantization_config)
    pass


if __name__ == "__main__":
    # try_peft()
    try_peft2()
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
