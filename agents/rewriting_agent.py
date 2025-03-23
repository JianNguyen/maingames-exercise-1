from typing import Annotated,Sequence

from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage, AIMessage
from langgraph.graph.message import add_messages
import google.generativeai as generativeai
import os

class RewritingAgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

class RewritingAgent:
    def __init__(self) -> None:
        generativeai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        self.generation_config = {
            "temperature": 1.0,
            "top_p": 0.95,
            "top_k": 60,
            "max_output_tokens": 8096,
        }
        self.llm_model = generativeai.GenerativeModel(

            model_name="gemini-2.0-flash",
            generation_config=self.generation_config
        )
        self.prompt = """ 
                You are an expert in conversational refinement, your task is to transform user inputs into standalone, context-free questions.

                **Instructions:**

                1.  **Rewrite:**  Rephrase the most recent user input into a complete, self-contained question, as if the user were asking it directly.
                2.  **Clarity:** Ensure the reformulated question is clear, unambiguous, and requires no prior conversation history to understand.
                3.  **Conciseness:** The rewritten question should be as brief as possible while still capturing the user's complete intent.
                4.  **No Assumptions:** Do not add any information not explicitly present in the user input, and do not ask follow-up questions.
                5.  **Output:** Provide ONLY the rewritten question as a single, grammatically correct sentence.

                DO NOT provide any additional context, explanations, or information beyond the rewritten question itself. DO NOT answer user question, just rewrite the question. 
                Your response must have same language of user inputs. KEEP SAME INTENT AS USER QUESTION.
                Do NOT answer the question; just provide the reformulated version. Do not translate English terms or technical words (e.g., laptop, TV, CPU, GPU) into other languages in your response.
                **YOU MUST TRANSLATE TO ENGLISH.**

                Here is user inputs:
                Conversational History:
                {history}
                User input:
                {user_input}
                """

    def __call__(self ,*args):
        self.state = args[0]
        messages = self.state["messages"]

        history = messages[-1].additional_kwargs.get("history")
        user_input = messages[-1].content
        self.prompt = self.prompt.format(history = history, user_input = user_input)
        response = self.llm_model.generate_content(self.prompt)
        response = {"messages": AIMessage(content=response.text.strip(),
                                          additional_kwargs={"history": history}
                                          )
                    }
        return response
