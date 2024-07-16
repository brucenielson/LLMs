from haystack import Pipeline, Document, component
from haystack.components.preprocessors import DocumentCleaner, DocumentSplitter
from haystack.components.embedders import SentenceTransformersDocumentEmbedder, SentenceTransformersTextEmbedder
from haystack_integrations.components.retrievers.pgvector import PgvectorEmbeddingRetriever
from haystack_integrations.document_stores.pgvector import PgvectorDocumentStore
from haystack.components.converters import HTMLToDocument
from haystack.dataclasses import ByteStream
from bs4 import BeautifulSoup
from ebooklib import epub, ITEM_DOCUMENT
from haystack.components.writers import DocumentWriter
from typing import List


@component
class RemoveIllegalDocs:
    """
    A component that removes duplicates or empty documents from a list of documents.
    """
    @component.output_types(documents=List[Document])
    def run(self, documents: List[Document]):
        # Remove documents with empty content
        documents = [Document(content=doc.content) for doc in documents if doc.content is not None]
        # Remove duplicate Documents with duplicates with duplicate document ids
        documents = list({doc.id: doc for doc in documents}.values())

        return {"documents": documents}


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


def doc_splitter_pipeline(document_store: PgvectorDocumentStore):
    pipeline = Pipeline()
    # https://docs.haystack.deepset.ai/docs/htmltodocument
    pipeline.add_component("converter", HTMLToDocument())
    pipeline.add_component("remove_duplicates", instance=RemoveIllegalDocs())
    # https://docs.haystack.deepset.ai/docs/documentcleaner
    pipeline.add_component("cleaner", DocumentCleaner())
    # https://docs.haystack.deepset.ai/docs/documentsplitter
    pipeline.add_component("splitter", DocumentSplitter(split_by="word",
                                                        split_length=400,
                                                        split_overlap=0,
                                                        split_threshold=100))

    # https://docs.haystack.deepset.ai/docs/sentencetransformersdocumentembedder
    pipeline.add_component("embedder", SentenceTransformersDocumentEmbedder())
    # Write out to the document store (PgvectorDocumentStore)
    pipeline.add_component("writer", DocumentWriter(document_store=document_store))

    # Connect the components
    pipeline.connect("converter", "remove_duplicates")
    pipeline.connect("remove_duplicates", "cleaner")
    pipeline.connect("cleaner", "splitter")
    pipeline.connect("splitter", "embedder")
    pipeline.connect("embedder", "writer")

    return pipeline


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
        # docs_with_embeddings = create_embeddings(split_docs)["documents"]
        # document_store.write_documents(docs_with_embeddings)
        pipeline = doc_splitter_pipeline(document_store)
        results = pipeline.run({"converter": {"sources": sources}})
        print("\n\n")
        print(f"Number of documents: {results['writer']['documents_written']}")

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
    document_store = initialize_and_load_documents(epub_file_path, recreate_table=False)
    query_pipeline = create_query_pipeline(document_store)
    query = "Are we a democracy or a republic?"
    result = query_pipeline.run({"text_embedder": {"text": query}})
    documents = result['retriever']['documents']
    for doc in documents:
        print(doc.content)
        print(f"Score: {doc.score}")
        print("")


main()
