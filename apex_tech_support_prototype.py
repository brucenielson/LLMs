from haystack_integrations.document_stores.pgvector import PgvectorDocumentStore
from haystack import Document
import json
# import os

# Initialize the PgvectorDocumentStore
document_store = PgvectorDocumentStore(
    table_name="haystack_docs",
    embedding_dimension=768,
    vector_function="cosine_similarity",
    recreate_table=True,
    search_strategy="hnsw",
)


def load_and_store_documents(file_path):
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
        document_store.write_documents(documents)


def load_and_print_first_entry(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        # Read the entire JSON file
        data = json.load(file)
        # Print the very first entry to understand its structure
        if data:
            print("First entry in the JSON file:")
            print(json.dumps(data[0], indent=2))


# Load and store documents from the JSON file
load_and_store_documents(r'D:\Projects\Holley\apex_data.json')
# load_and_print_first_entry(r'D:\Projects\Holley\apex_data.json')
