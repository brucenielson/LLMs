import google.generativeai as genai


def get_gemini_secret():
    secret_file = r'D:\Documents\Secrets\gemini_secret.txt'
    try:
        with open(secret_file, 'r') as file:
            secret_text = file.read()
    except FileNotFoundError:
        print(f"The file '{secret_file}' does not exist.")
    except Exception as e:
        print(f"An error occurred: {e}")

    return secret_text


def test_gemini():
    model = genai.GenerativeModel(model_name="gemini-pro")

    # Generate text using the prompt
    response = model.generate_content(prompt)
    # Print the generated text
    print("\n\n")
    print(response.text)
    print("\n\n")


key = get_gemini_secret()
genai.configure(api_key=key)
models = genai.list_models()
for model in models:
    print(model.name)
    print(model.description)

