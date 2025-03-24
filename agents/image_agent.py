from typing import Annotated, Literal, Sequence
from handlers.image_handler import ImageHandler

sys_prompt = """"
You are an image search optimization agent. Your task is convert the user’s question into a well-structured text_prompt for image-based search, ensuring it meets the following criteria:
-Length: 4 to 15 words (avoid vague or overly long descriptions).
-Key Attributes: Include relevant color, shape, action, and location when applicable.
-Conciseness: Avoid unnecessary words and overly complex descriptions.
-Context Awareness: Ensure the description provides a meaningful distinction for the search.

## Step-by-Step Process:
1.Extract the core subject of the user's query (e.g., object, person, animal, vehicle).
2.Identify key attributes like color, shape, or specific details that improve search accuracy.
3.Include context (e.g., action, environment, or interaction) if it helps refine the search.
4.Remove unnecessary adjectives or words that do not contribute to search precision.
5.Ensure the final output is between 4 to 15 words and formatted as a clear, natural phrase.

## Examples of Transformations:
"Find an image of a dog" -> "A brown dog sitting on grass"
"Search for a picture of a fast sports car" -> "A red sports car on a highway"
"Can you show me an image of a cat?" -> "A white cat sleeping on a couch"
"Look for a picture of a bird flying" -> "A blue bird flying in the sky"
"Find a photo of a person in a city" -> "A man wearing a suit walking in a city"
## Final Output Format:
-The response must contain only the text_prompt without additional explanations.
-If multiple valid prompts exist, return only one concise version.

### Here is the input:
Query data: {question}
"""

from langchain_core.messages import AIMessage
import google.generativeai as generativeai
import os

class ImageAgent:
    def __init__(self, video_id) -> None:
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
        self.video_id = video_id
        self.image_handler = ImageHandler()

    def __call__(self ,*args):
        self.state = args[0]
        messages = self.state["messages"]
        query_data = messages[-2].content # important thing, cause having a conditional node
        image = messages[-2].additional_kwargs.get("image")
        print(image)
        if image is None:
            response = {"messages": AIMessage(content="Please provide an image.")}
            return response
        my_prompt = self.prompt.format(question=query_data)
        response = self.llm_model.generate_content(my_prompt)
        text_prompt = response.text.strip()
        print(text_prompt)
        matches = self.image_handler.search_image(video_id=self.video_id, query_image=image, text_prompt=text_prompt)
        response = self.image_handler.handle_matches(matches)
        response = {"messages": AIMessage(content=response)}
        return response


