from typing import Annotated, Literal, Sequence


sys_prompt = """"
You are an information verification agent. Your task is to analyze word occurrences in audio and provide a concise, accurate report to the user.

INPUT:
- Search word: "word or phrase"
- Query data: "List of tuples with word, start time, and end time"

INSTRUCTIONS:
Follow these steps carefully to analyze the data and report your findings:

Step 1: Check if the search word appears in the query data.
   - Examine each word in the query data
   - Compare with the search word (consider case-insensitivity and close matches)
   - Note the number of exact matches and close matches

Step 2: Analyze the timing patterns.
   - Identify when the search word first appears (earliest timestamp)
   - Note if the word appears multiple times and their distribution
   - Calculate the total duration the word is spoken (sum of end_time - start_time)

Step 3: Evaluate the context.
   - Look for patterns in word frequency or clustering
   - Note if occurrences are evenly distributed or concentrated

Step 4: Generate a concise report with:
   - Confirmation of whether the search word appears in the data
   - Total number of occurrences
   - Timestamp of first appearance and any significant clusters
   - Brief assessment of the relevance/importance based on frequency and timing

OUTPUT FORMAT:
- Keep your response brief and focused on key findings
- Present timing information in a clear, readable format
- If the search word does not appear, clearly state this fact
- Include only complete, important insights - avoid partial thoughts
- Give opinion if there is any wrong or unreasonable complaint in the query data

Example Report:
"The word 'climate' appears 7 times. First occurrence at 2:15. Notable clusters at 5:30-6:45 (4 mentions). Based on frequency and clustering, this appears to be a central topic in the discussion."

Here is the input:
Search word: {search_word}
Query data: {query_data}
"""

from langchain_core.messages import BaseMessage, AIMessage
import google.generativeai as generativeai
import os

class WordsTimestampAnalyst:
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
        query_data = messages[-1].content
        search_word = messages[-1].additional_kwargs.get("search_word")
        print(query_data)
        print(search_word)
        my_prompt = self.prompt.format(query_data=query_data, search_word=search_word)
        response = self.llm_model.generate_content(my_prompt)
        response = {"messages": AIMessage(content=response.text.strip())}
        return response

