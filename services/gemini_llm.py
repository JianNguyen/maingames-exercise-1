import os
from google import genai
from google.genai import types
from langchain.text_splitter import RecursiveCharacterTextSplitter

from dotenv import load_dotenv
load_dotenv("../.env")


class GeminiLLM:
    def __init__(self):
        self.client = genai.Client(
            api_key=os.getenv("GEMINI_API_KEY"),
        )
        self.gen_model_name = "gemini-2.0-flash"
        self.embed_model_name = "models/text-embedding-004"
        self.generate_content_config = types.GenerateContentConfig(
            temperature=1,
            top_p=0.95,
            top_k=40,
            max_output_tokens=8192,
            response_mime_type="text/plain",
        )


    def generate(self, text):
        contents = [
            types.Content(
                role="user",
                parts=[
                    types.Part.from_text(text=text),
                ],
            ),
        ]

        final_text = "".join(chunk.text for chunk in self.client.models.generate_content_stream(
            model=self.gen_model_name,
            contents=contents,
            config=self.generate_content_config
        ))

        return final_text.strip()


    def embed(self, text):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,  # Max length of each chunk
            chunk_overlap=200  # Overlap between chunks
        )

        chunks = text_splitter.split_text(text)

        embedding_values = []
        for chunk in chunks:
            result = self.client.models.embed_content(
                model=self.embed_model_name,
                contents=chunk)

            embedding_values.append({
                "embedding": result.embeddings[0].values,
                "text_chunk": chunk
            })
        return embedding_values

    def get_embedding(self, text):
        result = self.client.models.embed_content(
            model=self.embed_model_name,
            contents=text)
        return result.embeddings[0].values

if __name__ == "__main__":
    llm_model = GeminiLLM()
    # print(llm_model.generate("hello"))
    print(llm_model.embed("hello, world"))
