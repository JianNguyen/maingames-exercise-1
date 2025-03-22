from services.gemini_llm import GeminiLLM
from services.pgvector.connector import PgVector


class ChatHandler:
    def __init__(self):
        self.llm_model = GeminiLLM()
        self.pg_vector = PgVector()

    def chat(self, user_input, video_id):
        query_vector = self.llm_model.get_embedding(user_input)
        final_context = ""
        contexts, additional_contexts = self.pg_vector.search_vector(video_id, query_vector)
        if contexts:
            for context in contexts:
                final_context += context[1] + "\n"
        response = self.llm_model.chatbot(context=final_context, question=user_input)

        return response


