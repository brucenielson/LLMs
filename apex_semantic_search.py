import json
import os
import torch
from sentence_transformers import SentenceTransformer, util
from huggingface_hub import login
import time
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer, util
from google.generativeai import GenerativeModel
import google.generativeai as genai


def generate_tech_support_response(query, model):
    # Load the embeddings or create and save them if not found
    embeddings_list = load_apex_embeddings()
    if embeddings_list is None:
        data = load_apex_data()
        embeddings_list = create_and_save_embeddings(data, model)

    # Perform semantic search
    top_results = semantic_search_top(embeddings_list, query, model, top=3)

    # Summarize the proposed resolutions
    summarized_cases = "\n\n".join([f"Case Id: {entry['CaseId']}\nProblem: {entry['Problem']}"
                                          f"\nResolution: {entry['Resolution']}\n\n" for entry in top_results])

    # Generate a tech support response using the language model
    model_id = "meta-llama/Llama-2-7b-hf"
    causal_model = AutoModelForCausalLM.from_pretrained(model_id).to("cuda")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    prompt = (f"As a tech support agent, the customer reports this problem:\n\n{query}\n\n"
              f"Similar problems in the past: \n"
              f"{summarized_cases}\n\n"
              f"Write a summary of past resolutions to similar problems:\n")
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to("cuda")

    start_time = time.time()
    output = causal_model.generate(input_ids, max_length=1000, num_return_sequences=1)
    end_time = time.time()

    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    print("CUDA execution time (ms):", (end_time - start_time) * 1000)
    print(generated_text)


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

def save_apex_embeddings(embeddings_list):
    # Save the embeddings list to D:\Projects\Holley\apex_embeddings.json
    file_path = r'D:\Projects\Holley\apex_embeddings.json'
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(embeddings_list, file)

def load_apex_data():
    # Load D:\Projects\Holley\Apex Copilot\bin\apex_data.json
    file_path = r'D:\Projects\Holley\Apex Copilot\bin\apex_data.json'
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data_list = json.load(file)
    except UnicodeDecodeError as e:
        print(f"UnicodeDecodeError: {e}")
        print(e)

    # Now, data_list contains the contents of the JSON file as a list
    return data_list

def create_and_save_embeddings(json_data, model):
    # Create embeddings for the data and save them
    model = model.to('cuda' if torch.cuda.is_available() else 'cpu')

    problems = [entry['Problem'] for entry in json_data]
    # Encode problems into embeddings
    problem_embeddings = model.encode(problems, convert_to_tensor=True)

    embeddings_list = [
        {
            'Problem': json_data[i]['Problem'],
            'CaseId': json_data[i]['CaseId'],
            'Resolution': json_data[i]['Resolution'],
            'Embedding': problem_embeddings[i].tolist(),
        }
        for i in range(len(problems))
    ]

    # Save the embeddings list to the file
    save_apex_embeddings(embeddings_list)

    return embeddings_list

def semantic_search_top(embeddings_list, query, model, top=5):
    # Load embeddings if not provided
    if embeddings_list is None:
        data = load_apex_data()
        model = model.to('cuda' if torch.cuda.is_available() else 'cpu')
        embeddings_list = create_and_save_embeddings(data, model)

    # Retrieve embeddings for query
    query_embedding = model.encode(query, convert_to_tensor=True).to('cuda' if torch.cuda.is_available() else 'cpu')

    # Calculate cosine similarities between query and problems
    # Calculate cosine similarities between query and problems
    similarities = [
        {
            'Index': i,
            'CaseId': entry['CaseId'],
            'Similarity': util.pytorch_cos_sim(query_embedding, torch.tensor(entry['Embedding']).to('cuda')).item(),
            'Problem': entry['Problem'],
            'Resolution': entry['Resolution']
        }
        for i, entry in enumerate(embeddings_list)
    ]

    # Get the indices of the top 5 matches
    top_cases = sorted(similarities, key=lambda k: k['Similarity'], reverse=True)[:top]
    top_indices = [result['Index'] for result in top_cases]

    # Print the top 5 matches with Problem, and Resolution
    print("Top Matching Resolutions:")
    for i, index in enumerate(top_indices, 1):
        result = similarities[index]
        print(f"{i}. Index: {result['Index']}, \n Case Id: {result['CaseId']}, \nProblem: {result['Problem']}, \nResolution: {result['Resolution']}\n\n")

    return top_cases

def generate_gemini_response(query, top_results):
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
    GOOGLE_API_KEY = secret_text
    genai.configure(api_key=GOOGLE_API_KEY)

    # Create a summary of past resolutions to similar problems
    summarized_cases = "\n\n".join([f"Case Id: {entry['CaseId']}\nProblem: {entry['Problem']}"
                                    f"\nResolution: {entry['Resolution']}\n\n" for entry in top_results])

    # Define the prompt including similar problems
    prompt = (f"As a tech support agent, the customer reports this problem:\n\n{query}\n\n"
              f"Similar problems in the past: \n"
              f"{summarized_cases}\n\n"
              f"Suggest possible resolutions based on similar problems:\n")

    # Initialize the Gemini model
    model = GenerativeModel(model_name="gemini-pro")

    # Generate text using the prompt
    response = model.generate_content(prompt)

    # Print the generated text
    print(response.text)

def query():
    # secret_file = r'D:\Projects\Holley\huggingface_secret.txt'
    # try:
    #     with open(secret_file, 'r') as file:
    #         secret_text = file.read()
    #         print(secret_text)
    # except FileNotFoundError:
    #     print(f"The file '{secret_file}' does not exist.")
    #     return
    # except Exception as e:
    #     print(f"An error occurred: {e}")
    #     return
    # login(secret_text)

    # Load the embeddings or create and save them if not found
    embeddings_list = load_apex_embeddings()
    model = SentenceTransformer('sentence-transformers/multi-qa-mpnet-base-dot-v1')
    if embeddings_list is None:
        data = load_apex_data()
        embeddings_list = create_and_save_embeddings(data, model)

    # query_text = 'Customer is trying to program the truck for the first time. He is getting an Error code 1006. Customer has never programmed before.'
    query_text = 'Idle on the truck goes up and down when unit is powered on and off.'
    model = SentenceTransformer('sentence-transformers/multi-qa-mpnet-base-dot-v1')
    # generate_tech_support_response(query_text, model)
    # Perform semantic search
    top_results = semantic_search_top(embeddings_list, query_text, model, top=5)
    # Generate Gemini response
    generate_gemini_response(query_text, top_results)


query()
