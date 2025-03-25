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
        chat_prompt = """You are a question-answering assistant designed to be extremely accurate. You *must* only answer questions based on the provided excerpt from a YouTube video transcript. Do not use any other information, even if you think you know the answer. It is very important that you do *not* hallucinate or make up information. If the answer is not explicitly stated in the transcript, you *must* say, "I am unable to answer that question based on the provided transcript." Be concise and factual.\n
            YouTube Transcript Excerpt:\n
            {context}\n
            Conversation History:\n
            {history}\n
            User Question: {question}\n
            Your Answer (based *solely* on the transcript):\n
            """
        my_prompt = chat_prompt.format(context=context, history=self.history, question=question)
        response = self.gen_model.generate_content(my_prompt).text.strip()
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
        sum_prompt = """"You are an advanced summarization agent designed to extract key information from a given text step by step. Your goal is to generate a structured summary that includes essential details while interleaving important keywords naturally. The summary should be optimized for user queries such as:
                    -"What is it about?"
                    -"What is the video discussing?"
                    -"What is mentioned in the video?"
                    ###Step-by-Step Instructions:
                    1.Identify the Main Topic:
                        -Extract the core subject of the video.
                        -Ensure the main topic appears early in the summary.
                    2.Highlight Key Points:
                        -Break the video content into main ideas and subtopics.
                        -Keep sentences concise and insert relevant keywords naturally.
                    3.Include Contextual Keywords:
                        -Ensure that keywords related to people, places, events, or important terms are naturally woven into the text.
                    4.Ensure Readability:
                        -Maintain clear, structured sentences for easy understanding.
                        -Avoid unnecessary complexity.
                    Example Output:
                    "The video is about Cristiano Ronaldo answering a series of yes or no questions, verified by a lie detector. It discusses his personal preferences, career achievements, and future aspirations. Throughout the interview, Ronaldo talks about his favorite number, goal-scoring records, and opinions on fast food, the Premier League, and Sir Alex Ferguson. This summary ensures that users can easily find details about Ronaldo’s confidence, ambitions, and the key moments covered in the video."
                    Here is the context:
                    {context}
                    """
        my_prompt = sum_prompt.format(context=context)
        response = self.gen_model.generate_content(my_prompt).text.strip()
        return response

if __name__ == "__main__":
    llm_model = GeminiLLM()
    # print(llm_model.generate("hello"))
    print(llm_model.embed("hello, world"))
