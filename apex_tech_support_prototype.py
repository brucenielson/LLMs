# Useful links (use for blog posts)
# https://haystack.deepset.ai/integrations/pgvector-documentstore
# https://github.com/pgvector/pgvector
# https://www.youtube.com/watch?v=Ff3tJ4pJEa4
# https://learn.microsoft.com/en-us/cpp/build/building-on-the-command-line?view=msvc-170#download-and-install-the-tools
# https://www.timescale.com/blog/how-to-install-psql-on-mac-ubuntu-debian-windows/
# https://minervadb.xyz/installing-and-configuring-pgvector-in-postgresql/
# https://phoenixnap.com/kb/install-postgresql-windows
# https://stackoverflow.com/questions/40183108/python-packages-hash-not-matching-whilst-installing-using-pip

# https://haystack.deepset.ai/integrations/pgvector-documentstore
# https://docs.haystack.deepset.ai/v2.0/docs/pgvectordocumentstore
# https://docs.haystack.deepset.ai/v2.0/docs/pgvectorembeddingretriever
# https://docs.haystack.deepset.ai/v2.0/docs/sentencetransformersdocumentembedder

#
# Doesn't work https://haystack.deepset.ai/tutorials/06_better_retrieval_via_embedding_retrieval
# from haystack.nodes import EmbeddingRetriever
# https://www.datastax.com/guides/hierarchical-navigable-small-worlds
# https://www.analyticsvidhya.com/blog/2023/10/introduction-to-hnsw-hierarchical-navigable-small-world/

# Docker
# https://docs.docker.com/desktop/install/windows-install/

# Crew AI
# https://www.crewai.com/
# https://x.com/akshay_pachaar/status/1793620669122617776


# https://medium.com/@cesarescalia/guide-haystack-deepset-qa-from-a-pdf-d3f83d76d9d2
# https://docs.haystack.deepset.ai/v1.15/docs/file_converters
# https://pymupdf.readthedocs.io/en/latest/
# https://pdfminersix.readthedocs.io/en/latest/


# https://haystack.deepset.ai/tutorials/16_document_classifier_at_index_time
# https://medium.com/@fvanlitsenburg/building-a-privategpt-with-haystack-part-1-why-and-how-de6fa43e18b
# https://haystack.deepset.ai/integrations/pgvector-documentstore#usage
# https://haystack.deepset.ai/integrations/pgvector-documentstore
# https://dev.to/stephenc222/how-to-use-postgresql-to-store-and-query-vector-embeddings-h4b
# https://docs.haystack.deepset.ai/docs/document-store
# https://docs.haystack.deepset.ai/docs/pgvectordocumentstore


from haystack_integrations.document_stores.pgvector import PgvectorDocumentStore
# from haystack import Document
import json
# from haystack.document_stores import DuplicatePolicy
from haystack import Document, Pipeline
from haystack.components.embedders import SentenceTransformersTextEmbedder, SentenceTransformersDocumentEmbedder
from haystack_integrations.components.retrievers.pgvector import PgvectorEmbeddingRetriever
from google.generativeai import GenerativeModel
import google.generativeai as genai
import textwrap
# Example of Haystack document that doesn't work!
# https://docs.haystack.deepset.ai/v1.15/docs/file_converters
# from haystack.nodes import PDFToTextConverter
from haystack.components.converters.pdfminer import PDFMinerToDocument
from datetime import datetime
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.preprocessors import DocumentCleaner
from haystack.components.preprocessors import DocumentSplitter

import re
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
                    "Resolution": entry["Resolution"],
                    "Source": 'Apex'
                }
            )
            documents.append(doc)
        return documents


def create_embeddings(documents):
    document_embedder = SentenceTransformersDocumentEmbedder()
    document_embedder.warm_up()
    documents_with_embeddings = document_embedder.run(documents)
    return documents_with_embeddings


def initialize_database(file_path, recreate_table=False):
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
        docs = []
        # Load the data from the Apex json
        apex_issues = load_documents(file_path)
        # apex_issues = []
        # Load the data from the User Guide
        user_guide = parse_pdf("85200_a.pdf", "CS/CTS User Guide")
        docs = user_guide + apex_issues
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


def generate_gemini_response(query_text, top_results):
    secret_file = r'D:\Projects\Holley\gemini_secret.txt'
    try:
        with open(secret_file, 'r') as file:
            secret_text = file.read()
    except FileNotFoundError:
        print(f"The file '{secret_file}' does not exist.")
        return
    except Exception as e:
        print(f"An error occurred: {e}")
        return

    # Configure the Gemini API with your API key
    key = secret_text
    genai.configure(api_key=key)

    # Create a summary of past resolutions to similar problems
    top_results_apex = [row for row in top_results[0:100] if row['Source'] == 'Apex'][:5]
    summarized_cases = "\n\n".join([f"Case Id: {entry['CaseId']}\nProblem: {entry['Problem']}"
                                    f"\nResolution: {entry['Resolution']}\n\n" for entry in top_results_apex])

    # Create manual text summary
    top_results_manual = [row for row in top_results if row['Source'] == 'PDF'][:2]
    manual_text = "\n\n".join([f"Source: {entry['Source Name']}\n\n{entry['Content']}" for entry in top_results_manual])

    if manual_text:
        manual_prompt = (
            f"Relevant Documentation:\n"
            f"(State 'Source' for anything used in your response.)"
            f"{manual_text}\n\n"
        )
    else:
        manual_prompt = None

    # Initialize the Gemini model
    model = GenerativeModel(model_name="gemini-pro")

    # if summarized_cases != "":
    #     # Define the prompt including similar problems
    #     prompt = (f"The customer reports this problem:\n\n{query_text}\n\n"
    #               f"Similar cases:\n"
    #               f"{summarized_cases}\n\n"
    #               f"Summarize in bullet points the similar cases and their problems and resolutions, "
    #               f"referencing them by Case Id:\n")
    #
    #     # Generate text using the prompt
    #     response = model.generate_content(prompt)
    #     # Print the generated text
    #     print("\n\n")
    #     print(response.text)
    #     print("\n\n")

    prompt = (f"The customer reports this problem:\n\n{query_text}\n\n"
              f"{manual_prompt}\n\n"
              f"Similar problems in the past:\n"
              f"{summarized_cases}\n\n"
              f"Suggest your top 2 to 5 possible resolutions based on similar problems. "
              f"Give a reference for each suggestion (either Case Ids or the manual 'Source' referenced) "
              f"in parentheses next to each recommendation. "
              f"You may reference more than one source in a proposed resolution if they "
              f"are similar resolutions.\n")
    # Generate text using the prompt
    response = model.generate_content(prompt)
    # Print the generated text
    print("Possible Resolutions:")
    print(response.text)
    print("\n\n")

    prompt = (f"The customer reports this problem:\n\n{query_text}\n\n"
              f"{manual_prompt}\n\n"
              f"Similar problems in the past:\n"
              f"{summarized_cases}\n\n"
              f"What do you recommend as a resolution? Answer in prose form explaining your thinking "
              f"by either referencing past similar problems (by Case Id) or by referencing the relevant manual.\n")
    # Generate text using the prompt
    response = model.generate_content(prompt)
    # Print the generated text
    print("My Recommendations:")
    print(textwrap.fill(response.text, width=80))
    print("\n\n")


def query(query_text, pipeline, top_k=5000):
    results = pipeline.run({"text_embedder": {"text": query_text}, "retriever": {"top_k": top_k}})
    documents = results['retriever']['documents']
    top_results = [{"Problem": doc.content, "Content": doc.content, **doc.meta} for doc in documents]
    # Generate Gemini response
    generate_gemini_response(query_text, top_results)


def menu(document_store):
    query_pipeline = create_haystack_pipeline(document_store)
    loop = True
    while loop:
        query_text = input("\n\nEnter customer's reported problem: ")
        if query_text.lower() == 'exit':
            loop = False
        else:
            query(query_text, query_pipeline)


def setup(reload_database=False):
    print("Starting database...")
    document_store = initialize_database(r'D:\Projects\Holley\apex_data.json', recreate_table=reload_database)
    print("Database initialized!")
    menu(document_store)


def clean_text(row):
    text = row.content
    # Replace newline with a space
    text = re.sub(r'\n', ' ', text)
    # Replace multiple consecutive spaces with a single space
    text = re.sub(r'\s+', ' ', text)
    # Remove all cases of '- '
    text = re.sub(r'- ', '', text)
    row.content = text
    return row


def parse_pdf(pdf_file, pdf_name=None):
    converter = PDFMinerToDocument()  # https://docs.haystack.deepset.ai/docs/pdfminertodocument
    result = converter.run(sources=[pdf_file],
                           meta={"date_added": datetime.now().isoformat(),
                                 "Source": 'PDF',
                                 "Source Name": pdf_name if pdf_name is not None else pdf_file})
    converted_document = result["documents"]
    # print(documents[0].content)
    cleaner = DocumentCleaner()  # https://docs.haystack.deepset.ai/docs/documentcleaner
    result = cleaner.run(documents=converted_document)
    cleaned_document = result["documents"]
    spliter = DocumentSplitter(split_by="sentence", split_length=8, split_overlap=5)  # https://docs.haystack.deepset.ai/docs/documentsplitter
    result = spliter.run(documents=cleaned_document)
    split_document = result["documents"]
    split_document = [clean_text(row) for row in split_document if not re.fullmatch(r'\.*', row.content)]
    return split_document


def test_pdf_pipeline():
    pipeline = Pipeline()
    pipeline.add_component("converter", PDFMinerToDocument())
    pipeline.add_component("cleaner", DocumentCleaner())
    pipeline.add_component("splitter", DocumentSplitter(split_by="sentence", split_length=5))
    return pipeline


# parse_pdf("85200_a.pdf", "CS/CTS User Guide")
setup(reload_database=False)
