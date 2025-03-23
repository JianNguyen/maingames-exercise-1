from agents.rewriting_agent import RewritingAgent, RewritingAgentState
from agents.answer_agent import AnswerAgent, AnswerAgentState
from agents.retrieval_agent import RetrievalAgent
from agents.detecting_demand_agent import DetectingQuestion
from agents.detecting_words_agent import DetectingWords
from agents.retrieval_timestamp_agent import RetrievalTimestampAgent
from langgraph.graph import StateGraph, START, END, MessagesState
from IPython.display import Image, display

from typing import Annotated,Sequence
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

class MultiAgentState:
    messages: Annotated[Sequence[BaseMessage], add_messages]

class MultipleAgents:

    def __init__(self,
                 video_id: str):
        self.video_id = video_id
        self.detecting_question = DetectingQuestion()
        self.rewriting_agent = RewritingAgent()
        self.retrieval_agent = RetrievalAgent(video_id=self.video_id)
        self.answer_agent = AnswerAgent()
        self.detecting_words = DetectingWords()
        self.retrieval_timestamp_agent = RetrievalTimestampAgent(video_id=self.video_id)
        self.graph = None


    def build(self):
        graph_builder = StateGraph(MessagesState)
        graph_builder.add_node("rewriting", self.rewriting_agent)
        graph_builder.add_node("detect_question", self.detecting_question)
        graph_builder.add_node("answer", self.answer_agent)
        graph_builder.add_node("personal", self.retrieval_agent)
        # for testing
        graph_builder.add_node("timestamp", self.detecting_words)
        graph_builder.add_node("timestamp_retrieval", self.retrieval_timestamp_agent)
        graph_builder.add_node("image", self.retrieval_agent)

        graph_builder.set_entry_point("rewriting")
        graph_builder.add_edge("rewriting", "detect_question")
        graph_builder.add_conditional_edges(
            "detect_question",
            self.detecting_question.routing)
        # route 1
        graph_builder.add_edge(
            "personal",
            "answer")
        graph_builder.add_edge(
            "answer",
            END)
        # route 2
        graph_builder.add_edge(
            "timestamp",
            "timestamp_retrieval")
        graph_builder.add_edge(
            "timestamp_retrieval",
            END)
        self.graph = graph_builder.compile()

    def display_graph(self):
        if self.graph is None:
            raise ValueError("You should build graph first!!!")

        img = Image(self.graph.get_graph().draw_mermaid_png())
        with open("architect.png", "wb") as png:
            png.write(img.data)

        return None