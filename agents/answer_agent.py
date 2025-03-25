from typing import Annotated,Sequence

from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage, AIMessage
from langgraph.graph.message import add_messages
import google.generativeai as generativeai
import os

sys_prompt = """
You are an advanced question-answering assistant designed to provide extremely accurate responses only based on the provided excerpt from a YouTube video transcript. Follow the step-by-step approach below to ensure precision and reliability.
## GUIDELINES
Step 1: Check for Available Data
If the transcript excerpt is empty or missing, respond with:
"Sorry, I cannot answer this question without more transcript data. Please provide more details."
If is available, proceed to Step 2.
Step 2: Understand the Question
Analyze question to determine what the user is asking.
Check if the requested information is explicitly stated in transcript.
Step 3: Extract the Answer
If the transcript clearly provides an answer, extract the most concise and factual response.
Avoid adding interpretations, assumptions, or external knowledge.
Step 4: Deliver the Response
Your final response should be:
-Concise – No unnecessary details.
-Factual – Directly based on the transcript.
-Accurate – No hallucination or added knowledge.

Here is the input:
Conversation history:
{history}
Transcript:
{context}
Question:
{question}
"""

class AnswerAgent:
    def __init__(self) -> None:
        generativeai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        self.generation_config = {
            "temperature": 1.0,
            "top_p": 0.95,
            "top_k": 64,
            "max_output_tokens": 8096,
        }
        self.llm_model = generativeai.GenerativeModel(
            model_name="gemini-2.0-pro-exp-02-05",
            generation_config=self.generation_config
        )
        self.prompt = sys_prompt

    def __call__(self ,*args):

        self.state = args[0]
        messages = self.state["messages"]
        history = messages[-1].additional_kwargs.get("history")
        user_input = messages[-1].additional_kwargs.get("user_input")
        context = messages[-1].content

        my_prompt = self.prompt.format(history=history, context=context, question=user_input)
        response = self.llm_model.generate_content(my_prompt)
        response = {"messages": AIMessage(content=response.text.strip())}
        return response
