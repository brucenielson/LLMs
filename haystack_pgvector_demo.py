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
    docs = []
    book = epub.read_epub(epub_file_path)

    # Find all paragraphs across sections
    for section in book.get_items_of_type(ITEM_DOCUMENT):
        section_html = section.get_body_content().decode('utf-8')
        section_soup = BeautifulSoup(section_html, 'html.parser')
        paragraphs = section_soup.find_all('p')
        byte_stream: ByteStream
        for p in paragraphs:
            p_str = str(p)
            p_html = f"<html><head><title>Converted Epub</title></head><body>{p_str}</body></html>"
            # https://docs.haystack.deepset.ai/docs/data-classes#bytestream
            byte_stream = ByteStream(p_html.encode('utf-8'))
            docs.append(byte_stream)
    return docs


# Initialize and load documents
def initialize_and_load_documents(epub_file_path, recreate_table=False):
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
        sources = load_epub(epub_file_path)

        # https://docs.haystack.deepset.ai/docs/htmltodocument
        converter = HTMLToDocument()
        results = converter.run(sources=sources)
        converted_docs = results["documents"]
        # Remove documents with empty content
        converted_docs = [Document(content=doc.content) for doc in converted_docs if doc.content is not None]
        # Remove duplicate Documents with duplicates with duplicate document ids
        converted_docs = list({doc.id: doc for doc in converted_docs}.values())

        # Clean the documents
        # https://docs.haystack.deepset.ai/docs/documentcleaner
        cleaner = DocumentCleaner()
        cleaned_docs = cleaner.run(documents=converted_docs)["documents"]

        # Split the documents
        # https://docs.haystack.deepset.ai/docs/documentsplitter
        splitter = DocumentSplitter(split_by="word",
                                    split_length=400,
                                    split_overlap=0,
                                    split_threshold=100)
        split_docs = splitter.run(documents=cleaned_docs)["documents"]
        docs_with_embeddings = create_embeddings(split_docs)["documents"]
        document_store.write_documents(docs_with_embeddings)

    return document_store


def create_embeddings(documents):
    document_embedder = SentenceTransformersDocumentEmbedder()
    document_embedder.warm_up()
    documents_with_embeddings = document_embedder.run(documents)
    return documents_with_embeddings


# Set up query pipeline
def create_query_pipeline(document_store):
    query_pipeline = Pipeline()
    query_pipeline.add_component("text_embedder", SentenceTransformersTextEmbedder())
    # https://docs.haystack.deepset.ai/docs/pgvectorembeddingretriever
    query_pipeline.add_component("retriever", PgvectorEmbeddingRetriever(document_store=document_store, top_k=5))
    query_pipeline.connect("text_embedder.embedding", "retriever.query_embedding")
    return query_pipeline


# Main function to run the semantic search
def main():
    epub_file_path = "Federalist Papers.epub"
    document_store = initialize_and_load_documents(epub_file_path)
    query_pipeline = create_query_pipeline(document_store)
    query = "What is the role of the judiciary in a democracy?"
    result = query_pipeline.run({"text_embedder": {"text": query}})
    documents = result['retriever']['documents']
    for doc in documents:
        print(doc.content)
        print(f"Score: {doc.score}")
        print("")


main()
