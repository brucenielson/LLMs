# Hugging Face and Pytorch imports
import torch
# Haystack imports
from haystack import Pipeline, Document, component
from haystack.components.embedders import SentenceTransformersDocumentEmbedder, SentenceTransformersTextEmbedder
from haystack.components.builders import PromptBuilder
from haystack.components.generators import HuggingFaceLocalGenerator
from haystack_integrations.components.generators.google_ai import GoogleAIGeminiGenerator
from haystack_integrations.components.retrievers.pgvector import PgvectorEmbeddingRetriever
from haystack_integrations.document_stores.pgvector import PgvectorDocumentStore
from haystack.utils import ComponentDevice, Device
from haystack.utils.auth import Secret
# Other imports
from typing import List, Optional, Dict, Any, Union
from pathlib import Path
import generator_model as gen


class RagPipeline:
    """
    A class that implements a Retrieval-Augmented Generation (RAG) system using Haystack and Pgvector.

    This class provides functionality to set up and use a RAG system for question answering
    tasks on a given corpus of text, currently from an EPUB file. It handles document
    indexing, embedding, retrieval, and generation of responses using a language model.

    The system uses a Postgres database with the Pgvector extension for efficient
    similarity search of embedded documents.

    Public Methods:
        draw_pipelines(): Visualize the RAG and document conversion pipelines.
        generate_response(query: str): Generate a response to a given query.

    Properties:
        sentence_context_length: Get the context length of the sentence embedder.
        sentence_embed_dims: Get the embedding dimensions of the sentence embedder.

    The class handles initialization of the document store, embedding models,
    and language models internally. It also manages the creation and execution
    of the document processing and RAG pipelines.
    """
    def __init__(self,
                 table_name: str = 'haystack_pgvector_docs',
                 postgres_user_name: str = 'postgres',
                 postgres_password: str = None,
                 postgres_host: str = 'localhost',
                 postgres_port: int = 5432,
                 postgres_db_name: str = 'postgres',
                 generator_model: Union[gen.GeneratorModel, HuggingFaceLocalGenerator, GoogleAIGeminiGenerator] = None,
                 embedder_model_name: Optional[str] = None,
                 ) -> None:
        """
        Initialize the HaystackPgvector instance.

        Args:
            table_name (str): Name of the table in the Pgvector database.
            postgres_user_name (str): Username for Postgres database.
            postgres_password (str): Password for Postgres database.
            postgres_host (str): Host address for Postgres database.
            postgres_port (int): Port number for Postgres database.
            postgres_db_name (str): Name of the Postgres database.
            generator_model (Union[GeneratorModel, HuggingFaceLocalGenerator, GoogleAIGeminiGenerator]):
                Language model to use for text generation.
            embedder_model_name (Optional[str]): Name of the embedding model to use.
        """

        # Instance variables
        self._table_name: str = table_name
        self._sentence_embedder: Optional[SentenceTransformersDocumentEmbedder] = None
        self._embedder_model_name: Optional[str] = embedder_model_name

        # GPU or CPU
        self._has_cuda: bool = torch.cuda.is_available()
        self._torch_device: torch.device = torch.device("cuda" if self._has_cuda else "cpu")
        self._component_device: ComponentDevice = ComponentDevice(Device.gpu() if self._has_cuda else Device.cpu())

        # Passwords and connection strings
        if postgres_password is None:
            raise ValueError("Postgres password must be provided")
        # PG_CONN_STR="postgresql://USER:PASSWORD@HOST:PORT/DB_NAME
        self._postgres_connection_str: str = (f"postgresql://{postgres_user_name}:{postgres_password}@"
                                              f"{postgres_host}:{postgres_port}/{postgres_db_name}")

        self._setup_embedder()

        print("Initializing document store")
        self._document_store: Optional[PgvectorDocumentStore] = None
        self._initialize_document_store()

        # Warm up _model
        if generator_model is None:
            raise ValueError("Generator model must be provided")
        self._generator_model: Union[gen.GeneratorModel, HuggingFaceLocalGenerator, GoogleAIGeminiGenerator] = (
            generator_model)
        # If the generator model has a warm_up method, call it
        if hasattr(self._generator_model, 'warm_up'):
            self._generator_model.warm_up()

        # Default prompt template
        # noinspection SpellCheckingInspection
        self._prompt_template: str = """
        <start_of_turn>user
        Quoting the information contained in the context where possible, give a comprehensive answer to the question.

        Context:
          {% for doc in documents %}
          {{ doc.content }}
          {% endfor %};

        Question: {{query}}<end_of_turn>

        <start_of_turn>model
        """

        # Declare rag pipeline
        self._rag_pipeline: Optional[Pipeline] = None
        # Create the RAG pipeline
        self._create_rag_pipeline()

    @property
    def sentence_context_length(self) -> Optional[int]:
        """
        Get the context length of the sentence embedder model.

        Returns:
            Optional[int]: The maximum context length of the sentence embedder model, if available.
        """
        self._setup_embedder()
        if self._sentence_embedder is not None and self._sentence_embedder.embedding_backend is not None:
            return self._sentence_embedder.embedding_backend.model.get_max_seq_length()
        else:
            return None

    @property
    def sentence_embed_dims(self) -> Optional[int]:
        """
        Get the embedding dimensions of the sentence embedder model.

        Returns:
            Optional[int]: The embedding dimensions of the sentence embedder model, if available.
        """
        self._setup_embedder()
        if self._sentence_embedder is not None and self._sentence_embedder.embedding_backend is not None:
            return self._sentence_embedder.embedding_backend.model.get_sentence_embedding_dimension()
        else:
            return None

    def _setup_embedder(self) -> None:
        if self._sentence_embedder is None:
            if self._embedder_model_name is not None:
                self._sentence_embedder = SentenceTransformersTextEmbedder(model=self._embedder_model_name,
                                                                           device=self._component_device,
                                                                           trust_remote_code=True)
            else:
                self._sentence_embedder = SentenceTransformersTextEmbedder(device=self._component_device)

            if hasattr(self._sentence_embedder, 'warm_up'):
                self._sentence_embedder.warm_up()

    def draw_pipeline(self) -> None:
        """
        Draw and save visual representations of the RAG and document conversion pipelines.
        """
        if self._rag_pipeline is not None:
            self._rag_pipeline.draw(Path("RAG Pipeline.png"))

    def generate_response(self, query: str) -> None:
        """
        Generate a response to a given query using the RAG pipeline.

        Args:
            query (str): The input query to process.
        """
        print("Generating Response...")
        results: Dict[str, Any] = self._rag_pipeline.run({
            "query_embedder": {"text": query},
            "prompt_builder": {"query": query}
        })

        merged_results = results["merger"]["merged_results"]

        # Print retrieved documents
        print("Retrieved Documents:")
        for i, doc in enumerate(merged_results["documents"], 1):
            print(f"Document {i}:")
            print(f"Score: {doc.score}")
            if hasattr(doc, 'meta') and doc.meta:
                if 'title' in doc.meta:
                    print(f"Title: {doc.meta['title']}")
                if 'section_num' in doc.meta:
                    print(f"Section: {doc.meta['section_num']}")
            print(f"Content: {doc.content}")
            print("-" * 50)

        # Print generated response
        # noinspection SpellCheckingInspection
        print("\nLLM's Response:")
        if merged_results["replies"]:
            answer: str = merged_results["replies"][0]
            print(answer)
        else:
            print("No response was generated.")

    @component
    class _MergeResults:
        @component.output_types(merged_results=Dict[str, Any])
        def run(self, documents: List[Document],
                replies: List[Union[str, Dict[str, str]]]) -> Dict[str, Dict[str, Any]]:
            return {
                "merged_results": {
                    "documents": documents,
                    "replies": replies
                }
            }

    def _initialize_document_store(self) -> None:
        connection_token: Secret = Secret.from_token(self._postgres_connection_str)
        document_store: PgvectorDocumentStore = PgvectorDocumentStore(
            connection_string=connection_token,
            table_name=self._table_name,
            embedding_dimension=self.sentence_embed_dims,
            vector_function="cosine_similarity",
            recreate_table=False,
            search_strategy="hnsw",
            hnsw_recreate_index_if_exists=True,
            hnsw_index_name=self._table_name + "_haystack_hnsw_index",
            keyword_index_name=self._table_name + "_haystack_keyword_index",
        )

        self._document_store = document_store
        print("Document Count: " + str(document_store.count_documents()))

    def _create_rag_pipeline(self) -> None:
        prompt_builder: PromptBuilder = PromptBuilder(template=self._prompt_template)

        rag_pipeline: Pipeline = Pipeline()
        # Use Cuda is possible
        rag_pipeline.add_component("query_embedder", self._sentence_embedder)
        rag_pipeline.add_component("retriever", PgvectorEmbeddingRetriever(document_store=self._document_store,
                                                                           top_k=5))
        rag_pipeline.add_component("prompt_builder", prompt_builder)
        if isinstance(self._generator_model, gen.GeneratorModel):
            rag_pipeline.add_component("llm", self._generator_model.generator_component)
        else:
            rag_pipeline.add_component("llm", self._generator_model)
        # Add a new component to merge results
        rag_pipeline.add_component("merger", self._MergeResults())

        rag_pipeline.connect("query_embedder.embedding", "retriever.query_embedding")
        rag_pipeline.connect("retriever.documents", "prompt_builder.documents")
        rag_pipeline.connect("prompt_builder", "llm")

        # Connect the retriever and llm to the merger
        rag_pipeline.connect("retriever.documents", "merger.documents")
        rag_pipeline.connect("llm.replies", "merger.replies")

        self._rag_pipeline = rag_pipeline


def main() -> None:
    postgres_password = gen.get_secret(r'D:\Documents\Secrets\postgres_password.txt')
    hf_secret: str = gen.get_secret(r'D:\Documents\Secrets\huggingface_secret.txt')  # Put your path here
    google_secret: str = gen.get_secret(r'D:\Documents\Secrets\gemini_secret.txt')  # Put your path here
    # model: gen.GeneratorModel = gen.HuggingFaceLocalModel(password=hf_secret, model_name="google/gemma-1.1-2b-it")
    # model: gen.GeneratorModel = gen.GoogleGeminiModel(password=google_secret)
    model: gen.GeneratorModel = gen.HuggingFaceAPIModel(password=hf_secret, model_name="HuggingFaceH4/zephyr-7b-alpha")
    rag_processor: RagPipeline = RagPipeline(table_name="federalist_papers",
                                             generator_model=model,
                                             postgres_user_name='postgres',
                                             postgres_password=postgres_password,
                                             postgres_host='localhost',
                                             postgres_port=5432,
                                             postgres_db_name='postgres',
                                             embedder_model_name="Alibaba-NLP/gte-large-en-v1.5")

    # Draw images of the pipelines
    rag_processor.draw_pipeline()
    print("Generator Embedder Dims: " + str(model.embedding_dimensions))
    print("Generator Context Length: " + str(model.context_length))
    print("Sentence Embedder Dims: " + str(rag_processor.sentence_embed_dims))
    print("Sentence Embedder Context Length: " + str(rag_processor.sentence_context_length))

    query: str = "What is the difference between a republic and a democracy?"
    rag_processor.generate_response(query)


if __name__ == "__main__":
    main()
