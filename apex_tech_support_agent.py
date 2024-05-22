from haystack_integrations.document_stores.pgvector import PgvectorDocumentStore
from haystack import Pipeline
from haystack.components.converters import TextFileToDocument
from haystack.components.writers import DocumentWriter
from haystack.components.embedders import SentenceTransformersDocumentEmbedder
import json
import os
from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack_integrations.components.retrievers.pgvector import PgvectorEmbeddingRetriever
from haystack_integrations.components.retrievers.pgvector import PgvectorKeywordRetriever


document_store = PgvectorDocumentStore(
    table_name="haystack_docs",
    embedding_dimension=768,
    vector_function="cosine_similarity",
    recreate_table=True,
    search_strategy="hnsw",
)


def load_apex_embeddings():
    # Load D:\Projects\Holley\apex_embeddings.json if it exists
    file_path = r'D:\Projects\Holley\apex_embeddings.json'
    if os.path.exists(file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                embeddings_list = json.load(file)
                return embeddings_list
        except UnicodeDecodeError as e:
            print(f"UnicodeDecodeError: {e}")
            print(e)
    return None


def create_haystack_pipeline():
    file_paths = [r'D:\Projects\Holley\apex_data.json']
    indexing = Pipeline()
    indexing.add_component("converter", TextFileToDocument())
    indexing.add_component("embedder", SentenceTransformersDocumentEmbedder())
    indexing.add_component("writer", DocumentWriter(document_store))
    indexing.connect("converter", "embedder")
    indexing.connect("embedder", "writer")
    indexing.run({"converter": {"sources": file_paths}})


def retrieve_from_doc_store():
    querying = Pipeline()
    querying.add_component("embedder", SentenceTransformersTextEmbedder())
    querying.add_component("retriever", PgvectorEmbeddingRetriever(document_store=document_store, top_k=3))
    querying.connect("embedder", "retriever")

    results = querying.run({"embedder": {"text": "my query"}})


def retrieve_key_word_match():
    retriever = PgvectorKeywordRetriever(document_store=document_store, top_k=3)
    results = retriever.run(query="my query")

# create_haystack_pipeline()
