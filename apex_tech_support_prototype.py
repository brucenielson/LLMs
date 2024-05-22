from haystack_integrations.document_stores.pgvector import PgvectorDocumentStore
from haystack import Document
import json
from haystack.components.embedders import SentenceTransformersDocumentEmbedder
# import os


def load_and_store_documents(ds, file_path):
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
        ds.write_documents(documents)


def load_and_print_first_entry(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        # Read the entire JSON file
        data = json.load(file)
        # Print the very first entry to understand its structure
        if data:
            print("First entry in the JSON file:")
            print(json.dumps(data[0], indent=2))


def create_embeddings(ds):
    print(f"Found {ds.count_documents()} documents in the store.")
    # Retrieve all documents using filter_documents (no filter criteria)
    documents = ds.filter_documents()


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
    # Load the database from the json if not already loaded
    if ds.count_documents() == 0:
        load_and_store_documents(ds, file_path)
    # Document database now initialized - embed database if necessary
    create_embeddings(ds)
    return ds


document_store = initialize_database(r'D:\Projects\Holley\apex_data.json')
# load_and_print_first_entry(r"D:\Projects\Holley\apex_data.json")
