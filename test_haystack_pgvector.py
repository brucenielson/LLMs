from haystack_integrations.document_stores.pgvector import PgvectorDocumentStore
from haystack import Document


def initialize_database(recreate_table=False):
    # Create a connection string in this format:
    # PG_CONN_STR="postgresql://USER:PASSWORD@HOST:PORT/DB_NAME
    # Initialize the PgvectorDocumentStore
    ds = PgvectorDocumentStore(
        table_name="haystack_docs",
        embedding_dimension=768,
        vector_function="cosine_similarity",
        recreate_table=recreate_table,
        search_strategy="hnsw",
        hnsw_recreate_index_if_exists=True
    )
    if ds.count_documents() == 0:
        # Do loading of documents
        pass
    return ds


def test_initialize_database():
    ds = initialize_database(recreate_table=True)
    ds.write_documents([
        Document(content="This is first", embedding=[0.1] * 768),
        Document(content="This is second", embedding=[0.3] * 768)
    ])
    print(ds.count_documents())


test_initialize_database()
