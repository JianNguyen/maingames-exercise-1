from typing import Annotated,Sequence

from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage, AIMessage
from langgraph.graph.message import add_messages
import google.generativeai as generativeai
import os

sys_prompt = """
You are a question refinement agent. Your task is to analyze a user's question along with their conversation history and rewrite their question to capture their exact intention.

INPUT:
- User's current question: {question}
- Conversation history:\n
{history}

INSTRUCTIONS:
Follow these steps carefully to analyze the user's intent and rewrite their question:

Step 1: Understand the explicit question.
   - Identify the core information the user is asking for
   - Note any specific terms, names, or concepts mentioned
   - Look for constraints or parameters the user has specified

Step 2: Analyze the conversation history.
   - Review previous exchanges for context
   - Identify any recurring themes or interests
   - Note any previous questions that this new question builds upon
   - Look for clarifications or refinements the user has made to earlier questions

Step 3: Identify implicit intentions.
   - Determine what problem the user is trying to solve
   - Consider what underlying goal the user might have
   - Recognize assumptions that may not be explicitly stated

Step 4: Rewrite the question to:
   - Be precise and unambiguous
   - Include all relevant context from the conversation history
   - Preserve the user's exact terminology and technical language
   - Maintain the user's core intent while adding clarity
   - Include specific parameters or constraints needed for a complete answer

OUTPUT FORMAT:
Return ONLY the rewritten question without any additional explanation, analysis, or commentary.

IMPORTANT:
- Preserve the user's exact terminology
- Do not introduce new concepts not implied by the user
- Do not change the fundamental intent of the question
- If the original question is already clear and complete, make only minor refinements
- Focus on precision and clarity rather than simply rewording

"""

class RewritingAgent:
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
        self.prompt = sys_prompt

    def __call__(self ,*args):
        self.state = args[0]
        messages = self.state["messages"]
        additional_kwargs = messages[-1].additional_kwargs
        history = additional_kwargs.get("history")
        user_input = messages[-1].content

        history = self.handle_history(history)
        print("Question: ", user_input)
        my_prompt = self.prompt.format(history=history, question=user_input)
        response = self.llm_model.generate_content(my_prompt)
        print("Rewriting agent :", response.text.strip())
        response = {"messages": AIMessage(content=response.text.strip(),
                                          additional_kwargs=additional_kwargs
                                          )
                    }
        return response

    def handle_history(self, history):
        result = ""
        for his in history:
            result += f'{his["role"]}: {his["content"]}\n '
        return result
