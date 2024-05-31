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

# import os

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


def generate_gemini_response(query_text, top_results_issues):
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
    summarized_cases = "\n\n".join([f"Case Id: {entry['CaseId']}\nProblem: {entry['Problem']}"
                                    f"\nResolution: {entry['Resolution']}\n\n" for entry in top_results_issues])

    # if top_results_manual:
    #     summarized_manual = "\n".join([f"{entry}" for entry in top_results_manual])
    #     manual_prompt = (
    #         f"Here is what is found in the users guide:\n"
    #         f"{summarized_manual}\n\n"
    #     )
    # else:
    #     manual_prompt = None

    # Define the prompt including similar problems
    prompt = (f"The customer reports this problem:\n\n{query_text}\n\n"
              f"Similar cases:\n"
              f"{summarized_cases}\n\n"
              f"Summarize in bullet points the similar cases and their problems and resolutions, "
              f"referencing them by Case Id:\n")

    # Initialize the Gemini model
    model = GenerativeModel(model_name="gemini-pro")

    # Generate text using the prompt
    response = model.generate_content(prompt)
    # Print the generated text
    print("\n\n")
    print(response.text)
    print("\n\n")

    prompt = (f"The customer reports this problem:\n\n{query_text}\n\n"
              # f"{manual_prompt}\n\n"
              f"Similar problems in the past:\n"
              f"{summarized_cases}\n\n"
              f"Suggest your top 2 to 5 possible resolutions based on similar problems. "
              f"Reference Case Ids and/or page of the user's manual in parentheses next to each recommendation. "
              f"You may reference more than one case id (or page of user's manual) in a proposed resolution if they "
              f"are similar resolutions.\n")
    # Generate text using the prompt
    response = model.generate_content(prompt)
    # Print the generated text
    print("Possible Resolutions:")
    print(response.text)
    print("\n\n")

    prompt = (f"The customer reports this problem:\n\n{query_text}\n\n"
              # f"{manual_prompt}\n\n"
              f"Similar problems in the past:\n"
              f"{summarized_cases}\n\n"
              f"What do you recommend as a resolution? Answer in prose form explaining your thinking "
              f"by referencing past similar problems by Case Id.\n")
    # Generate text using the prompt
    response = model.generate_content(prompt)
    # Print the generated text
    print("My Recommendations:")
    print(textwrap.fill(response.text, width=80))
    print("\n\n")


def query(query_text, pipeline, top_k=5):
    results = pipeline.run({"text_embedder": {"text": query_text}, "retriever": {"top_k": top_k}})
    documents = results['retriever']['documents']
    top_results = [{"Problem": doc.content, **doc.meta} for doc in documents]
    # Generate Gemini response
    generate_gemini_response(query_text, top_results)
    # for doc in documents:
    #     print(doc.content)
    #     print(doc.meta['Resolution'])
    #     print()


def menu(document_store):
    query_pipeline = create_haystack_pipeline(document_store)
    loop = True
    while loop:
        query_text = input("\n\nEnter customer's reported problem: ")
        if query_text.lower() == 'exit':
            loop = False
        else:
            query(query_text, query_pipeline)


def setup():
    print("Starting database...")
    document_store = initialize_database(r'D:\Projects\Holley\apex_data.json')
    print("Database initialized!")
    menu(document_store)


setup()
