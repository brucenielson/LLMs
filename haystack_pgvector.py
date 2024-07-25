from typing import List, Optional, Dict, Any, Tuple
from bs4 import BeautifulSoup
from ebooklib import epub, ITEM_DOCUMENT
import torch
import huggingface_hub as hf_hub
from pathlib import Path

from haystack import Pipeline, Document, component
from haystack.dataclasses import ByteStream
from haystack.components.preprocessors import DocumentCleaner, DocumentSplitter
from haystack.components.embedders import SentenceTransformersDocumentEmbedder, SentenceTransformersTextEmbedder
from haystack.components.converters import HTMLToDocument
from haystack.components.writers import DocumentWriter
from haystack.components.builders import PromptBuilder
from haystack.components.generators import HuggingFaceLocalGenerator
from haystack_integrations.components.retrievers.pgvector import PgvectorEmbeddingRetriever
from haystack_integrations.document_stores.pgvector import PgvectorDocumentStore
from haystack.utils import ComponentDevice, Device
from haystack.document_stores.types import DuplicatePolicy
from haystack.utils.auth import Secret
from transformers import AutoConfig, AutoTokenizer, PreTrainedTokenizerFast


class HaystackPgvector:
    def __init__(self,
                 table_name: str = 'document_store',
                 recreate_table: bool = False,
                 book_file_path: Optional[str] = None,
                 hf_password: Optional[str] = None,
                 postgres_user_name: str = 'postgres',
                 postgres_password: str = None,
                 postgres_host: str = 'localhost',
                 postgres_port: int = 5432,
                 postgres_db_name: str = 'postgres',
                 llm_model_name: str = 'google/gemma-1.1-2b-it',
                 text_embedder_model_name: Optional[str] = None,
                 min_section_size: int = 1000,
                 ) -> None:

        # Passwords and connection strings
        if hf_password is not None:
            hf_hub.login(hf_password, add_to_git_credential=False)
        self.postgres_user_name = postgres_user_name
        self.postgres_password = postgres_password
        self.postgres_host = postgres_host
        self.postgres_port = postgres_port
        self.postgres_db_name = postgres_db_name

        self.text_embedder_model_name: Optional[str] = text_embedder_model_name
        print("Starting up Text Embedder")
        self.sentence_embedder: SentenceTransformersTextEmbedder
        if self.text_embedder_model_name is not None:
            self.sentence_embedder = SentenceTransformersTextEmbedder(
                model_name_or_path=self.text_embedder_model_name)
        else:
            self.sentence_embedder = SentenceTransformersTextEmbedder()
        self.sentence_embedder.warm_up()

        self.book_file_path: Optional[str] = book_file_path
        self.table_name: str = table_name
        self.recreate_table: bool = recreate_table
        self.min_section_size = min_section_size

        print("Initializing document store")
        self.document_store: Optional[PgvectorDocumentStore] = None
        self._initialize_document_store()

        # Warm up generator
        self.has_cuda: bool = torch.cuda.is_available()
        self.torch_device: torch.device = torch.device("cuda" if self.has_cuda else "cpu")
        self.component_device: Device = Device.gpu() if self.has_cuda else Device.cpu()
        print("Starting up Large Language Model")
        self.llm_model_name: str = llm_model_name
        self.generator: HuggingFaceLocalGenerator = HuggingFaceLocalGenerator(
            model=self.llm_model_name,
            task="text-generation",
            device=ComponentDevice(self.component_device),
            generation_kwargs={
                "max_new_tokens": 500,
                "temperature": 0.6,
                "do_sample": True,
            })
        self.generator.warm_up()

        # Default prompt template
        # noinspection SpellCheckingInspection
        self.prompt_template: str = """
        <start_of_turn>user
        Quoting the information contained in the context where possible, give a comprehensive answer to the question.

        Context:
          {% for doc in documents %}
          {{ doc.content }}
          {% endfor %};

        Question: {{query}}<end_of_turn>

        <start_of_turn>model
        """
        # Declare pipelines
        self.rag_pipeline: Optional[Pipeline] = None
        self.doc_convert_pipeline: Optional[Pipeline] = None
        # Create the RAG pipeline
        self._create_rag_pipeline()
        # Save off a tokenizer
        # self.tokenizer: PreTrainedTokenizerFast = AutoTokenizer.from_pretrained(self.llm_model_name)

    @property
    def llm_context_length(self) -> Optional[int]:
        return HaystackPgvector._get_context_length(self.llm_model_name)

    @property
    def llm_embed_dims(self) -> Optional[int]:
        return HaystackPgvector._get_embedding_dimensions(self.llm_model_name)

    @property
    def sentence_context_length(self) -> Optional[int]:
        return HaystackPgvector._get_context_length(self.sentence_embedder.model)

    @property
    def sentence_embed_dims(self) -> Optional[int]:
        if self.sentence_embedder is not None and self.sentence_embedder.embedding_backend is not None:
            return self.sentence_embedder.embedding_backend.model.get_sentence_embedding_dimension()
        else:
            return None

    @component
    class RemoveIllegalDocs:
        @component.output_types(documents=List[Document])
        def run(self, documents: List[Document]) -> Dict[str, List[Document]]:
            documents = [Document(content=doc.content, meta=doc.meta) for doc in documents if doc.content is not None]
            documents = list({doc.id: doc for doc in documents}.values())
            return {"documents": documents}

    @component
    class MergeResults:
        @component.output_types(merged_results=Dict[str, Any])
        def run(self, documents: List[Document], replies: List[str]) -> Dict[str, Dict[str, Any]]:
            return {
                "merged_results": {
                    "documents": documents,
                    "replies": replies
                }
            }

    def _load_epub(self) -> Tuple[List[ByteStream], List[Dict[str, str]]]:
        docs: List[ByteStream] = []
        meta: List[Dict[str, str]] = []
        book: epub.EpubBook = epub.read_epub(self.book_file_path)
        section_num: int = 1
        for i, section in enumerate(book.get_items_of_type(ITEM_DOCUMENT)):
            section_html: str = section.get_body_content().decode('utf-8')
            section_soup: BeautifulSoup = BeautifulSoup(section_html, 'html.parser')
            headings = [heading.get_text().strip() for heading in section_soup.find_all('h1')]
            title = ' '.join(headings)
            paragraphs: List[Any] = section_soup.find_all('p')
            temp_docs: List[ByteStream] = []
            temp_meta: List[Dict[str, str]] = []
            total_text: str = ""
            for p in paragraphs:
                p_str: str = str(p)
                # Concatenate paragraphs to form a single document string
                total_text += p_str
                p_html: str = f"<html><head><title>Converted Epub</title></head><body>{p_str}</body></html>"
                byte_stream: ByteStream = ByteStream(p_html.encode('utf-8'))
                meta_node: Dict[str, str] = {"section_num": section_num, "title": title}
                temp_docs.append(byte_stream)
                temp_meta.append(meta_node)

            # If the total text length is greater than the minimum section size, add the section to the list
            if len(total_text) > self.min_section_size:
                docs.extend(temp_docs)
                meta.extend(temp_meta)
                section_num += 1
        return docs, meta

    def draw_pipelines(self) -> None:
        if self.rag_pipeline is not None:
            self.rag_pipeline.draw(Path("RAG Pipeline.png"))
        if self.doc_convert_pipeline is not None:
            self.doc_convert_pipeline.draw(Path("Document Conversion Pipeline.png"))

    def _doc_converter_pipeline(self) -> None:
        doc_convert_pipe: Pipeline = Pipeline()
        doc_convert_pipe.add_component("converter", HTMLToDocument())
        doc_convert_pipe.add_component("remove_illegal_docs", instance=self.RemoveIllegalDocs())
        doc_convert_pipe.add_component("cleaner", DocumentCleaner())
        doc_convert_pipe.add_component("splitter", DocumentSplitter(split_by="sentence", split_length=10,
                                                                    split_overlap=1,
                                                                    split_threshold=2))
        doc_convert_pipe.add_component("embedder", SentenceTransformersDocumentEmbedder())
        doc_convert_pipe.add_component("writer",
                                       DocumentWriter(document_store=self.document_store,
                                                      policy=DuplicatePolicy.OVERWRITE))

        doc_convert_pipe.connect("converter", "remove_illegal_docs")
        doc_convert_pipe.connect("remove_illegal_docs", "cleaner")
        doc_convert_pipe.connect("cleaner", "splitter")
        doc_convert_pipe.connect("splitter", "embedder")
        doc_convert_pipe.connect("embedder", "writer")

        self.doc_convert_pipeline = doc_convert_pipe

    def _initialize_document_store(self) -> None:
        if (self.postgres_password is None) or (self.postgres_password == ""):
            self.postgres_password = HaystackPgvector.get_secret(r'D:\Documents\Secrets\postgres_password.txt')
        # PG_CONN_STR="postgresql://USER:PASSWORD@HOST:PORT/DB_NAME
        connection_str: str = (f"postgresql://{self.postgres_user_name}:{self.postgres_password}@"
                               f"{self.postgres_host}:{self.postgres_port}/{self.postgres_db_name}")
        connection_token: Secret = Secret.from_token(connection_str)
        document_store: PgvectorDocumentStore = PgvectorDocumentStore(
            connection_string=connection_token,
            table_name=self.table_name,
            embedding_dimension=self.sentence_embed_dims,
            vector_function="cosine_similarity",
            recreate_table=self.recreate_table,
            search_strategy="hnsw",
            hnsw_recreate_index_if_exists=True,
            hnsw_index_name=self.table_name+"_haystack_hnsw_index",
            keyword_index_name=self.table_name+"_haystack_keyword_index",
        )

        self.document_store = document_store

        if document_store.count_documents() == 0 and self.book_file_path is not None:
            sources: List[ByteStream]
            meta: List[Dict[str, str]]
            print("Loading document file")
            sources, meta = self._load_epub()
            print("Writing documents to document store")
            self._doc_converter_pipeline()
            results: Dict[str, Any] = self.doc_convert_pipeline.run({"converter": {"sources": sources, "meta": meta}})
            print(f"\n\nNumber of documents: {results['writer']['documents_written']}")

    def _create_rag_pipeline(self) -> None:
        prompt_builder: PromptBuilder = PromptBuilder(template=self.prompt_template)

        rag_pipeline: Pipeline = Pipeline()
        rag_pipeline.add_component("query_embedder", SentenceTransformersTextEmbedder())
        rag_pipeline.add_component("retriever", PgvectorEmbeddingRetriever(document_store=self.document_store,
                                                                           top_k=5))
        rag_pipeline.add_component("prompt_builder", prompt_builder)
        rag_pipeline.add_component("llm", self.generator)
        # Add a new component to merge results
        rag_pipeline.add_component("merger", self.MergeResults())

        rag_pipeline.connect("query_embedder.embedding", "retriever.query_embedding")
        rag_pipeline.connect("retriever.documents", "prompt_builder.documents")
        rag_pipeline.connect("prompt_builder.prompt", "llm.prompt")

        # Connect the retriever and llm to the merger
        rag_pipeline.connect("retriever.documents", "merger.documents")
        rag_pipeline.connect("llm.replies", "merger.replies")

        self.rag_pipeline = rag_pipeline

    def generative_response(self, query: str) -> None:
        print("Generating Response...")
        results: Dict[str, Any] = self.rag_pipeline.run({
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

    @staticmethod
    def _get_context_length(model_name: str) -> Optional[int]:
        config: AutoConfig = AutoConfig.from_pretrained(model_name)
        context_length: Optional[int] = getattr(config, 'max_position_embeddings', None)
        if context_length is None:
            context_length = getattr(config, 'n_positions', None)
        if context_length is None:
            context_length = getattr(config, 'max_sequence_length', None)
        return context_length

    @staticmethod
    def _get_embedding_dimensions(model_name: str) -> Optional[int]:
        # TODO: Need to test if this really gives us the embedder dims.
        #  Works correctly for SentenceTransformersTextEmbedder
        config: AutoConfig = AutoConfig.from_pretrained(model_name)
        embedding_dims: Optional[int] = getattr(config, 'hidden_size', None)
        return embedding_dims

    @staticmethod
    def get_secret(secret_file: str) -> str:
        try:
            with open(secret_file, 'r') as file:
                secret_text: str = file.read().strip()
        except FileNotFoundError:
            print(f"The file '{secret_file}' does not exist.")
            secret_text = ""
        except Exception as e:
            print(f"An error occurred: {e}")
            secret_text = ""

        return secret_text


def main() -> None:
    secret: str = HaystackPgvector.get_secret(r'D:\Documents\Secrets\huggingface_secret.txt')

    epub_file_path: str = "Federalist Papers.epub"
    processor: HaystackPgvector = HaystackPgvector(table_name="federalist_papers",
                                                   recreate_table=False,
                                                   book_file_path=epub_file_path,
                                                   hf_password=secret)

    # Draw images of the pipelines
    processor.draw_pipelines()
    print("LLM Embedding Dims: " + str(processor.llm_embed_dims))
    print("LLM Context Length: " + str(processor.llm_context_length))
    print("Sentence Embedding Dims: " + str(processor.sentence_embed_dims))
    print("Sentence Context Length: " + str(processor.sentence_context_length))

    query: str = "What is the difference between a republic and a democracy?"
    processor.generative_response(query)


if __name__ == "__main__":
    main()