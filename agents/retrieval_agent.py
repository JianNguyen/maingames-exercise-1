from services.gemini_llm import GeminiLLM
from services.pgvector.connector import PgVector
from langchain_core.messages import AIMessage


class RetrievalAgent:
    def __init__(self,video_id):
        self.pg_vector = PgVector()
        self.llm_model = GeminiLLM()
        self.video_id = video_id

    def __call__(self ,*args):
        self.state = args[0]
        messages = self.state["messages"]
        user_input = messages[-2].content # important thing, cause having a conditional node
        history = messages[-1].additional_kwargs.get("history")
        response = self.search_vector(user_input=user_input)
        response = {"messages": AIMessage(content=response.strip(),
                                          additional_kwargs={"history": history,
                                                             "user_input": user_input
                                                             }
                                          )
                    }
        return response

    def search_vector(self, user_input):
        query_vector = self.llm_model.get_embedding(user_input)
        final_context = ""
        contexts, additional_contexts = self.pg_vector.search_vector(self.video_id, query_vector, distance_threshold=0.5)
        if contexts:
            for context in contexts:
                final_context += context[1] + "\n"
        if additional_contexts:
            for additional_context in additional_contexts:
                final_context += additional_context[0] + "\n"

        return final_context
