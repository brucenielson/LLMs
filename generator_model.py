# Hugging Face and Pytorch imports
import torch
import huggingface_hub as hf_hub
from transformers import AutoConfig
# Haystack imports
from haystack.components.generators import HuggingFaceLocalGenerator
from haystack_integrations.components.generators.google_ai import GoogleAIGeminiGenerator
from haystack.components.generators import HuggingFaceAPIGenerator
from haystack.utils import ComponentDevice, Device
from haystack.utils.auth import Secret
# Other imports
from typing import Optional, Union
from abc import ABC, abstractmethod


def get_secret(secret_file: str) -> str:
    """
    Read a secret from a file.

    Args:
        secret_file (str): Path to the file containing the secret.

    Returns:
        str: The content of the secret file, or an empty string if an error occurs.
    """
    try:
        with open(secret_file, 'r') as file:
            secret_text: str = file.read().strip()
    except FileNotFoundError:
        print(f"The file '{secret_file}' does not exist.")
        secret_text = ""
    except Exception as e:
        print(f"An error occurred: {e}")
        raise e

    return secret_text


class GeneratorModel(ABC):
    """
    A class that represents a Large Language Model (LLM) generator.

    This class provides functionality to generate text using a large language model
    that allows any supported model to have a similar interface for text generation.
    This allows a Hugging Face model and a Google AI model to be used interchangeably.
    In the future I may add additional options for other language models.

    Public Methods:
        generate(prompt: str): Generate text using the given prompt.

    The class handles the initialization of the language model and the generation
    of text using the model internally. It also manages the configuration of the
    generation parameters.
    """
    def __init__(self, verbose: bool = False) -> None:
        """
        Initialize the GeneratorModel instance.

        Args:
            verbose (bool, optional): Whether to print verbose output. Defaults to False.
        """
        self._verbose: bool = verbose
        if self._verbose:
            print("Warming up Large Language Model")

        self._model_name: Optional[str] = None
        self._model: Optional[Union[HuggingFaceLocalGenerator, GoogleAIGeminiGenerator, HuggingFaceAPIModel]] = None
        self._verbose: bool = verbose

    @property
    def generator_component(self) -> Union[HuggingFaceLocalGenerator, HuggingFaceAPIGenerator, GoogleAIGeminiGenerator]:
        """
        Get the generator component of the language model.

        Returns:
            Union[HuggingFaceLocalGenerator, GoogleAIGeminiGenerator]: The generator component of the language model
        """
        return self._model

    @property
    @abstractmethod
    def context_length(self) -> Optional[int]:
        """
        Get the generator component of the language model.

        Returns:
            Union[HuggingFaceLocalGenerator, GoogleAIGeminiGenerator]: The generator component of the language model
        """
        pass

    @property
    @abstractmethod
    def embedding_dimensions(self) -> Optional[int]:
        """
        Get the embedding dimensions of the language model.

        Returns:
            Optional[int]: The embedding dimensions of the language model, if available. Otherwise, returns None.
        """
        pass

    @property
    @abstractmethod
    def language_model(self) -> Optional[object]:
        """
        Get the language model instance.

        Returns:
            Returns the language model instance used by this generator. If it is an API model, returns None.
        """
        pass

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """
        Generate text using the given prompt.

        Args:
            prompt (str): The prompt to use for text generation.

        Returns:
            str: The generated text.
        """
        # To be implemented in a subclass
        pass

    @property
    def model_name(self) -> Optional[str]:
        return self._model_name

    @property
    def verbose(self) -> bool:
        return self._verbose

    @verbose.setter
    def verbose(self, value: bool) -> None:
        self._verbose = value


class HuggingFaceModel(GeneratorModel, ABC):
    def __init__(self,
                 model_name: str = 'google/gemma-1.1-2b-it',
                 max_new_tokens: int = 500,
                 temperature: float = 0.6,
                 password: Optional[str] = None,
                 verbose: bool = False) -> None:
        """
        Initialize the HuggingFaceModel instance.

        Args:
            model_name (str, optional): Name of the Hugging Face model to use. Defaults to 'google/gemma-1.1-2b-it'.
            max_new_tokens (int, optional): Maximum number of new tokens to generate. Defaults to 500.
            temperature (float, optional): Temperature for text generation. Defaults to 0.6.
            password (Optional[str], optional): Password for Hugging Face authentication. Defaults to None.
            verbose (bool, optional): Whether to print verbose output. Defaults to False.
        """
        super().__init__(verbose)

        self._max_new_tokens: int = max_new_tokens
        self._temperature: float = temperature
        self._model_name: str = model_name

        if password is not None:
            hf_hub.login(password, add_to_git_credential=False)

    @property
    def max_new_tokens(self) -> int:
        return self._max_new_tokens

    @property
    def temperature(self) -> float:
        return self._temperature

    @property
    def context_length(self) -> Optional[int]:
        try:
            config: AutoConfig = AutoConfig.from_pretrained(self._model_name)
        except Exception as e:
            print(f"An error occurred: {e}")
            return None
        context_length: Optional[int] = getattr(config, 'max_position_embeddings', None)
        if context_length is None:
            context_length = getattr(config, 'n_positions', None)
        if context_length is None:
            context_length = getattr(config, 'max_sequence_length', None)
        return context_length

    @property
    def embedding_dimensions(self) -> Optional[int]:
        # TODO: Need to test if this really gives us the embedder dims.
        #  Works correctly for SentenceTransformersTextEmbedder
        try:
            config: AutoConfig = AutoConfig.from_pretrained(self._model_name)
        except Exception as e:
            print(f"An error occurred: {e}")
            return None
        embedding_dims: Optional[int] = getattr(config, 'hidden_size', None)
        return embedding_dims

    def generate(self, prompt: str) -> str:
        return self._model.run(prompt)


class HuggingFaceLocalModel(HuggingFaceModel):
    """
    A class that represents a Hugging Face Large Language Model (LLM) generator.

    """

    def __init__(self,
                 model_name: str = 'google/gemma-1.1-2b-it',
                 max_new_tokens: int = 500,
                 temperature: float = 0.6,
                 password: Optional[str] = None,
                 task: str = "text-generation",
                 verbose: bool = True) -> None:
        """
        Initialize the HuggingFaceLocalModel instance.

        Args:
            model_name (str, optional): Name of the language model to use. Defaults to 'google/gemma-1.1-2b-it'.
            max_new_tokens (int, optional): Maximum number of new tokens to generate. Defaults to 500.
            temperature (float, optional): Temperature for text generation. Defaults to 0.6.
            password (Optional[str], optional): Password for Hugging Face authentication. Defaults to None.
            task (str, optional): The task to perform using the language model. Defaults to "text-generation".
            verbose (bool, optional): Whether to print verbose output. Defaults to True.
        """
        super().__init__(verbose=verbose, model_name=model_name, max_new_tokens=max_new_tokens,
                         temperature=temperature, password=password)

        # Local model related variables
        self._task: str = task
        self._has_cuda: bool = torch.cuda.is_available()
        self._torch_device: torch.device = torch.device("cuda" if self._has_cuda else "cpu")
        self._component_device: ComponentDevice = ComponentDevice(Device.gpu() if self._has_cuda else Device.cpu())

        self._model: HuggingFaceLocalGenerator = HuggingFaceLocalGenerator(
            model=self._model_name,
            task="text-generation",
            device=self._component_device,
            generation_kwargs={
                "max_new_tokens": self._max_new_tokens,
                "temperature": self._temperature,
                "do_sample": True,
            })

    def warm_up(self) -> None:
        self._model.warm_up()

    def language_model(self) -> object:
        return self._model.pipeline.model


class HuggingFaceAPIModel(HuggingFaceModel):
    """
    A class that represents a Hugging Face Large Language Model (LLM) generator.

    """

    def __init__(self,
                 model_name: str = 'google/gemma-1.1-2b-it',
                 max_new_tokens: int = 500,
                 password: Optional[str] = None,
                 temperature: float = 0.6,
                 verbose: bool = True) -> None:
        """
        Initialize the GeneratorModel instance.

        Args:
            model_name (str): Name of the language model to use.
        """
        super().__init__(verbose=verbose, model_name=model_name, max_new_tokens=max_new_tokens,
                         temperature=temperature, password=password)

        self._max_new_tokens: int = max_new_tokens
        self._temperature: float = temperature
        self._model_name: str = model_name

        self._model: HuggingFaceAPIGenerator = HuggingFaceAPIGenerator(
            api_type="serverless_inference_api",
            api_params={
                "model": self._model_name,
            },
            token=Secret.from_token(password),
            generation_kwargs={
                "max_new_tokens": self._max_new_tokens,
                "temperature": self._temperature,
                "do_sample": True,
            })

    def generate(self, prompt: str) -> str:
        return self._model.run(prompt)

    def language_model(self) -> None:
        return None


class GoogleGeminiModel(GeneratorModel):
    """
    A class that represents a Google AI Large Language Model (LLM) generator.

    """

    def __init__(self, password: Optional[str] = None) -> None:
        """
        Initialize the GoogleGeminiModel instance.

        Args:
            password (Optional[str], optional): Password for Google AI authentication. Defaults to None.
        """
        super().__init__()
        if self._verbose:
            print("Warming up Large Language Model")

        self._model: GoogleAIGeminiGenerator = GoogleAIGeminiGenerator(
            model="gemini-pro",
            api_key=Secret.from_token(password)
        )

    def generate(self, prompt: str) -> str:
        return self._model.run(prompt)

    @property
    def context_length(self) -> Optional[int]:
        return None

    @property
    def embedding_dimensions(self) -> Optional[int]:
        return None

    @property
    def language_model(self) -> None:
        return None
