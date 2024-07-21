import os
from typing import List
from bs4 import BeautifulSoup
from ebooklib import epub, ITEM_DOCUMENT
import torch
from huggingface_hub import login

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


class HaystackPgvectorDemo:
    def __init__(self, table_name: str = 'document_store', recreate_table: bool = False, book_file_path: str = None):
        # Attempt to login to Hugging Face
        secret = HaystackPgvectorDemo.get_secret()
        login(secret, add_to_git_credential=True)
        # os.environ["HF_API_TOKEN"] = secret

        self.book_file_path = book_file_path
        self.table_name = table_name
        self.recreate_table = recreate_table
        self.document_store = None
        self._initialize_document_store()
        self.has_cuda = torch.cuda.is_available()
        self.torch_device = torch.device("cuda" if self.has_cuda else "cpu")
        self.component_device = Device.gpu() if self.has_cuda else Device.cpu()
        self.rag_pipeline = self._create_rag_pipeline()

    @component
    class RemoveIllegalDocs:
        @component.output_types(documents=List[Document])
        def run(self, documents: List[Document]):
            documents = [Document(content=doc.content) for doc in documents if doc.content is not None]
            # Removes duplicates, but this is redundant because we also use DuplicatePolicy.OVERWRITE in DocumentWriter
            documents = list({doc.id: doc for doc in documents}.values())
            return {"documents": documents}

    def _load_epub(self):
        docs = []
        book = epub.read_epub(self.book_file_path)
        for section in book.get_items_of_type(ITEM_DOCUMENT):
            section_html = section.get_body_content().decode('utf-8')
            section_soup = BeautifulSoup(section_html, 'html.parser')
            paragraphs = section_soup.find_all('p')
            for p in paragraphs:
                p_str = str(p)
                p_html = f"<html><head><title>Converted Epub</title></head><body>{p_str}</body></html>"
                byte_stream = ByteStream(p_html.encode('utf-8'))
                docs.append(byte_stream)
        return docs

    def _doc_converter_pipeline(self):
        doc_convert_pipe = Pipeline()
        doc_convert_pipe.add_component("converter", HTMLToDocument())
        doc_convert_pipe.add_component("remove_illegal_docs", instance=self.RemoveIllegalDocs())
        doc_convert_pipe.add_component("cleaner", DocumentCleaner())
        doc_convert_pipe.add_component("splitter", DocumentSplitter(split_by="word", split_length=400,
                                                                    split_overlap=0,
                                                                    split_threshold=100))
        doc_convert_pipe.add_component("embedder", SentenceTransformersDocumentEmbedder())
        doc_convert_pipe.add_component("writer",
                                       DocumentWriter(document_store=self.document_store,
                                                      policy=DuplicatePolicy.OVERWRITE))

        doc_convert_pipe.connect("converter", "remove_illegal_docs")
        doc_convert_pipe.connect("remove_illegal_docs", "cleaner")
        doc_convert_pipe.connect("cleaner", "splitter")
        doc_convert_pipe.connect("splitter", "embedder")
        doc_convert_pipe.connect("embedder", "writer")

        return doc_convert_pipe

    def _initialize_document_store(self):
        # TODO: calculate embedding dimension
        # self.model.get_sentence_embedding_dimension()
        document_store = PgvectorDocumentStore(
            table_name=self.table_name,
            embedding_dimension=768,
            vector_function="cosine_similarity",
            recreate_table=self.recreate_table,
            search_strategy="hnsw",
            hnsw_recreate_index_if_exists=True
        )

        self.document_store = document_store

        if document_store.count_documents() == 0:
            sources = self._load_epub()
            pipeline = self._doc_converter_pipeline()
            results = pipeline.run({"converter": {"sources": sources}})
            print(f"\n\nNumber of documents: {results['writer']['documents_written']}")

    def _create_query_pipeline(self):
        query_pipeline = Pipeline()
        query_pipeline.add_component("text_embedder", SentenceTransformersTextEmbedder())
        query_pipeline.add_component("retriever",
                                     PgvectorEmbeddingRetriever(document_store=self.document_store, top_k=5))
        query_pipeline.connect("text_embedder.embedding", "retriever.query_embedding")
        return query_pipeline

    def _create_rag_pipeline(self):
        model_name = 'google/gemma-1.1-2b-it'

        generator = HuggingFaceLocalGenerator(
            model=model_name,
            task="text-generation",
            device=ComponentDevice(self.component_device),
            generation_kwargs={
                "max_new_tokens": 500,
                "temperature": 0.6,
                "do_sample": True,
            })
        generator.warm_up()

        prompt_template = """
        <start_of_turn>user
        Quoting the information contained in the context where possible, give a comprehensive answer to the question.

        Context:
          {% for doc in documents %}
          {{ doc.content }}
          {% endfor %};

        Question: {{query}}<end_of_turn>

        <start_of_turn>model
        """
        prompt_builder = PromptBuilder(template=prompt_template)

        rag = Pipeline()
        rag.add_component("query_embedder", SentenceTransformersTextEmbedder())
        rag.add_component("retriever", PgvectorEmbeddingRetriever(document_store=self.document_store, top_k=5))
        rag.add_component("prompt_builder", prompt_builder)
        rag.add_component("llm", generator)

        rag.connect("query_embedder.embedding", "retriever.query_embedding")
        rag.connect("retriever.documents", "prompt_builder.documents")
        rag.connect("prompt_builder.prompt", "llm.prompt")
        return rag

    def get_generative_answer(self, query: str):
        results = self.rag_pipeline.run({
            "query_embedder": {"text": query},
            "prompt_builder": {"query": query}
        })
        answer = results["llm"]["replies"][0]
        print(answer)

    @staticmethod
    def get_secret():
        secret_file = r'D:\Documents\Secrets\huggingface_secret.txt'
        try:
            with open(secret_file, 'r') as file:
                secret_text = file.read().strip()
        except FileNotFoundError:
            print(f"The file '{secret_file}' does not exist.")
        except Exception as e:
            print(f"An error occurred: {e}")

        return secret_text


def main():
    epub_file_path = "Federalist Papers.epub"
    processor = HaystackPgvectorDemo(table_name="federalist_papers",
                                     recreate_table=False,
                                     book_file_path=epub_file_path)

    query = "What is the difference between a republic and a democracy?"
    processor.get_generative_answer(query)


if __name__ == "__main__":
    main()
