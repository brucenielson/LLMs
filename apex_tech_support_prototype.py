from haystack_integrations.document_stores.pgvector import PgvectorDocumentStore
from haystack import Document
import json
# from haystack.document_stores import DuplicatePolicy
from haystack import Document, Pipeline
from haystack.components.embedders import SentenceTransformersTextEmbedder, SentenceTransformersDocumentEmbedder
from haystack_integrations.components.retrievers.pgvector import PgvectorEmbeddingRetriever

# import os


def load_documents(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        # Read the entire JSON file
        data = json.load(file)
        documents = []
        for entry in data:
            doc = Document(
                content=entry["Problem"],
                meta={
                    "CaseId": entry["CaseId"],
                    "CreationDateTime": entry["CreationDateTime"],
                    "Resolution": entry["Resolution"]
                }
            )
            documents.append(doc)
        return documents


def load_and_print_first_entry(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        # Read the entire JSON file
        data = json.load(file)
        # Print the very first entry to understand its structure
        if data:
            print("First entry in the JSON file:")
            print(json.dumps(data[0], indent=2))


def create_embeddings(documents):
    document_embedder = SentenceTransformersDocumentEmbedder()
    document_embedder.warm_up()
    documents_with_embeddings = document_embedder.run(documents)
    return documents_with_embeddings


def initialize_database(file_path):
    # Initialize the PgvectorDocumentStore
    ds = PgvectorDocumentStore(
        table_name="haystack_docs",
        embedding_dimension=768,
        vector_function="cosine_similarity",
        recreate_table=False,
        search_strategy="hnsw",
        hnsw_recreate_index_if_exists=True
    )
    if ds.count_documents() == 0:
        # Load the database from the json if not already loaded
        docs = load_documents(file_path)
        # Document database now initialized - embed database if necessary
        docs_with_embeddings = create_embeddings(docs)
        ds.write_documents(docs_with_embeddings['documents'])
    return ds


def create_haystack_pipeline(ds):
    query_pipeline = Pipeline()
    query_pipeline.add_component("text_embedder", SentenceTransformersTextEmbedder())
    query_pipeline.add_component("retriever", PgvectorEmbeddingRetriever(document_store=ds))
    query_pipeline.connect("text_embedder.embedding", "retriever.query_embedding")
    return query_pipeline


document_store = initialize_database(r'D:\Projects\Holley\apex_data.json')
documents = document_store.filter_documents()
print(documents[0])
pipeline = create_haystack_pipeline(document_store)
query = ('Customer is trying to program the truck for the first time. He is getting an Error code 1006. '
         'Customer has never programmed before.')
result = pipeline.run({"text_embedder": {"text": query}})
documents = result['retriever']['documents']
for doc in documents:
    print(doc.content)
    print(doc.meta['Resolution'])
    print()

# load_and_print_first_entry(r"D:\Projects\Holley\apex_data.json")