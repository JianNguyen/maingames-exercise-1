import os
from google import genai
import google.generativeai as generativeai
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationChain


class GeminiLLM:
    def __init__(self):
        generativeai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        self.client = genai.Client(
            api_key=os.getenv("GEMINI_API_KEY"),
        )
        self.generation_config = {
            "temperature": 1,
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": 8192,
            "response_mime_type": "text/plain",
        }
        self.gen_model_name = "gemini-2.0-flash"
        self.gen_model = generativeai.GenerativeModel(
                            model_name=self.gen_model_name,
                            generation_config=self.generation_config
                            )
        self.embed_model_name = "models/text-embedding-004"

        self.history = []

    def add_to_history(self, user_message, bot_response, k=5):
        self.history.append({"user": user_message, "bot": bot_response})
        if len(self.history) > k:
            self.history.pop(0)

    def chatbot(self, context, question):
        prompt = """You are a question-answering assistant designed to be extremely accurate. You *must* only answer questions based on the provided excerpt from a YouTube video transcript. Do not use any other information, even if you think you know the answer. It is very important that you do *not* hallucinate or make up information. If the answer is not explicitly stated in the transcript, you *must* say, "I am unable to answer that question based on the provided transcript." Be concise and factual.\n
            YouTube Transcript Excerpt:\n
            {context}\n
            Conversation History:\n
            {history}\n
            User Question: {question}\n
            Your Answer (based *solely* on the transcript):\n
            """
        prompt = prompt.format(context=context, history=self.history, question=question)
        response = self.gen_model.generate_content(prompt).text.strip()
        self.add_to_history(question, response)
        return response


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

    def summarize(self, context):
        prompt = """You are a smart summarization agent. A user has provided some text from transcript of video and wants to know what it's about. Provide a clear, concise, and search-engine-friendly summary, adhering strictly to the following guidelines:\n
                    # Guidelines:\n
                    1.The answer must follow the following structure:\n
                        This is a summary of the video or youtube video: summarize the content of the text, summarize the main ideas and how they are linked together\n
                    2.Write without giving your own opinion in response, such as: Okay, here's the breakdown of the provided text, In my opinion,...\n
                    3.Use a professional tone: Your writing should be precise, formal, and well-structured throughout.\n
                    4.Optimized for Retrieval: Structure your response in a way that facilitates effective indexing and searchability in retrieval systems.\n
                    Think step-by-step, ensuring each sentence contributes to a complete and accurate summary.\n
                    Answer by the text:\n
                    {context}
                    """
        prompt = prompt.format(context=context)
        response = self.gen_model.generate_content(prompt).text.strip()
        print(response)
        return response

if __name__ == "__main__":
    llm_model = GeminiLLM()
    # print(llm_model.generate("hello"))
    print(llm_model.embed("hello, world"))
