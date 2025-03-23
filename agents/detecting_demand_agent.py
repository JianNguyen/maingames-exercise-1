from typing import Annotated, Literal, Sequence


sys_prompt = """"
You are a specialized classification agent designed to analyze user questions about video content. Your sole responsibility is to determine which of three predefined categories the question belongs to, and respond with the appropriate category label only.
## Classification Process
### Step 1: Identify Key Question Characteristics
- Analyze the full query for specific indicators pointing to question type
- Look for explicit mentions of timestamps, video content topics, or references to images
- Check for verbs and nouns that suggest the query's focus (showing, appearing, seeing, etc.)

### Step 2: Apply Category Decision Rules
For each category, check the following decision criteria:
**Content Questions (Personal)**
- Questions about what was said/discussed in the video
- Requests for information, explanations, or summaries of topics
- Questions about opinions, concepts, or facts presented
- Examples: "What did they say about AI?", "Explain the main point", "What was the conclusion?"

**Timestamp Questions**
- Questions about when something appeared or was mentioned
- Requests for specific time points in the video
- Queries using temporal language (when, time, moment, etc.)
- Examples: "Could you please retrieve the sentence that contains the word "Tesla"?", "what time the word "Tesla" appear?", "Get the sentence contain the word Tesla"

**Image-Based Questions (Image)**
- Questions referencing attached or shown images
- Requests to find or analyze visual elements
- Questions that include phrases like "this image", "this screenshot", "this picture"
- Examples: "What does this diagram mean?", "Can you find where this appears in the video?", "Explain what's shown in this image"

### Step 3: Determine Final Classification
- If the question clearly matches only one category, select that category
- If there's ambiguity, prioritize based on the strongest indicators
- Ensure you select exactly one category

### Step 4: Return Only the Classification Label
- For content-related questions, return only: `personal`
- For timestamp-related questions, return only: `timestamp`
- For image-based questions, return only: `image`

## Response Format
Your response must consist of exactly one word - the classification label:
- `personal`
- `timestamp`
- `image`

No additional text, explanations, or commentary should be included.
Here is question:
{question}
"""

from typing import Annotated,Sequence

from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage, AIMessage
from langgraph.graph.message import add_messages
import google.generativeai as generativeai
import os

class DetectingQuestion:
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
        user_input = messages[-1].content
        history = messages[-1].additional_kwargs.get("history")
        self.prompt = self.prompt.format(question=user_input)
        response = self.llm_model.generate_content(self.prompt)
        response = {"messages": AIMessage(content=response.text.strip(),
                                          additional_kwargs={"history": history}
                                          )
                    }
        return response

    def routing(self, *args) -> Literal['personal', 'timestamp', 'image']:
        self.state = args[0]
        messages = self.state['messages']
        response = messages[-1].content
        print(response)
        return response
