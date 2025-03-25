from handlers.speech2text import WhisperTranscriptor
from services.gemini_llm import GeminiLLM
from services.pgvector.connector import PgVector
import os

class VectorHandler:
    def __init__(self):
        self.arc_model = WhisperTranscriptor()
        self.llm_model = GeminiLLM()
        self.pg_vector = PgVector()
