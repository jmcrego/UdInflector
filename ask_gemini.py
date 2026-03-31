import os
import google.genai as genai
from google.genai import types


api_key=os.environ.get("GEMINI_API_KEY")
model = "gemini-flash-lite-latest"
client = genai.Client(api_key=api_key)
STREAMING = False

def generate(question):

    prompt = f"""You are a helpful assistant that answers questions.

    - Answer using the language of the question unless instructed otherwise.
    - Do not add comments or explanations, 
    - Use text without any formatting (no markdown),
    - Just answer the question directly using the minimum number of words.
    
    Q: {question}
    A:"""

    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(text=prompt),
            ],
        ),
    ]
    generate_content_config = types.GenerateContentConfig() #disallows completely thinking


    if STREAMING:
        for chunk in client.models.generate_content_stream(model=model, contents=contents, config=generate_content_config):
            print(chunk.text, end="")
            print()
    else:
        response = client.models.generate_content(model=model, contents=contents, config=generate_content_config)
        print(response.text, end="")
        print()

if __name__ == "__main__":
    while True:
        question = input("Enter your question: ")
        generate(question)