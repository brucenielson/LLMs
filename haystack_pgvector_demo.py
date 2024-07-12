import os
from haystack import Pipeline, Document
from haystack.components.preprocessors import TextCleaner, DocumentCleaner, DocumentSplitter
from haystack.components.embedders import SentenceTransformersDocumentEmbedder, SentenceTransformersTextEmbedder
from haystack_integrations.components.retrievers.pgvector import PgvectorEmbeddingRetriever
from haystack_integrations.document_stores.pgvector import PgvectorDocumentStore
from haystack.components.converters import HTMLToDocument
from haystack.dataclasses import ByteStream
from bs4 import BeautifulSoup
from ebooklib import epub, ITEM_DOCUMENT


def load_epub(epub_file_path):
    book = epub.read_epub(epub_file_path)
    html_content = ""

    # Find all paragraphs across sections
    for section in book.get_items_of_type(ITEM_DOCUMENT):
        section_html = section.get_body_content().decode('utf-8')
        section_soup = BeautifulSoup(section_html, 'html.parser')
        paragraphs = section_soup.find_all('p')
        html_content += ''.join([str(p) for p in paragraphs])  # Combine paragraph strings

    # Add missing elements (optional)
    html_content = f"<html><head><title>Converted Epub</title></head><body>{html_content}</body></html>"

    html_bytes = html_content.encode('utf-8')
    html_byte_stream = ByteStream(html_bytes)

    return html_byte_stream


# Initialize and load documents
def initialize_and_load_documents(epub_file_path, recreate_table=True):
    document_store = PgvectorDocumentStore(
        table_name="federalist_papers",
        embedding_dimension=768,
        vector_function="cosine_similarity",
        recreate_table=recreate_table,
        search_strategy="hnsw",
        hnsw_recreate_index_if_exists=True
    )

    if document_store.count_documents() == 0:
        # Convert EPUB to text documents
        document = load_epub(epub_file_path)

        # Convert bytes to a list (required by the converter)
        sources = [document]

        converter = HTMLToDocument()
        results = converter.run(sources=sources)

        # Clean the documents
        cleaner = DocumentCleaner()
        cleaned_docs = cleaner.run(documents=results["documents"])

        # Split the documents
        splitter = DocumentSplitter(
            split_by="word",
            split_length=200,
            split_overlap=20,
        )
        split_docs = splitter.run(documents=cleaned_docs["documents"])

        document_store.write_documents(split_docs["documents"])

    return document_store


# Set up indexing pipeline
def setup_indexing_pipeline():
    embedder = SentenceTransformersDocumentEmbedder(model_name="sentence-transformers/multi-qa-mpnet-base-dot-v1")

    indexing_pipeline = Pipeline()
    indexing_pipeline.add_component("embedder", embedder)
    return indexing_pipeline


# Run indexing pipeline
def run_indexing_pipeline(indexing_pipeline, document_store):
    documents = document_store.get_all_documents()
    results = indexing_pipeline.run({"documents": documents})
    document_store.write_documents(results["documents"])


# Set up query pipeline
def setup_query_pipeline(document_store):
    text_embedder = SentenceTransformersTextEmbedder(model_name="sentence-transformers/multi-qa-mpnet-base-dot-v1")
    retriever = PgvectorEmbeddingRetriever(document_store=document_store)

    query_pipeline = Pipeline()
    query_pipeline.add_component("text_embedder", text_embedder)
    query_pipeline.add_component("retriever", retriever)

    query_pipeline.connect("text_embedder.embedding", "retriever.query_embedding")

    return query_pipeline


# Main function to run the semantic search
def main():
    epub_file_path = "Federalist Papers.epub"
    document_store = initialize_and_load_documents(epub_file_path)

    indexing_pipeline = setup_indexing_pipeline()
    run_indexing_pipeline(indexing_pipeline, document_store)

    query_pipeline = setup_query_pipeline(document_store)


main()
