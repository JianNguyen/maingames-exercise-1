from typing import Annotated,Sequence

from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage, AIMessage
from langgraph.graph.message import add_messages
import google.generativeai as generativeai
import os

class AnswerAgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

class AnswerAgent:
    def __init__(self) -> None:
        generativeai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        self.generation_config = {
            "temperature": 1.0,
            "top_p": 0.95,
            "top_k": 45,
            "max_output_tokens": 8096,
        }
        self.llm_model = generativeai.GenerativeModel(
            model_name="gemini-2.0-flash",
            generation_config=self.generation_config
        )
        self.prompt = """You are a question-answering assistant designed to be extremely accurate. You *must* only answer questions based on the provided excerpt from a YouTube video transcript. Do not use any other information, even if you think you know the answer. It is very important that you do *not* hallucinate or make up information. If the answer is not explicitly stated in the transcript, you *must* say, "I am unable to answer that question based on the provided transcript." Be concise and factual.\n
                    YouTube Transcript Excerpt:\n
                    {context}\n
                    Conversation History:\n
                    {history}\n
                    User Question: {question}\n
                    Your Answer (based *solely* on the transcript):\n
                    """

    def __call__(self ,*args):

        self.state = args[0]
        messages = self.state["messages"]
        history = messages[-1].additional_kwargs.get("history")
        user_input = messages[-1].additional_kwargs.get("user_input")
        context = messages[-1].content

        self.prompt = self.prompt.format(history=history, context=context, question=user_input)
        response = self.llm_model.generate_content(self.prompt)
        response = {"messages": AIMessage(content=response.text.strip())}
        return response
