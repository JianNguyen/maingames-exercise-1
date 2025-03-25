from services.gemini_llm import GeminiLLM
from services.pgvector.connector import PgVector
from langchain_core.messages import AIMessage


class RetrievalTimestampAgent:
    def __init__(self,video_id):
        self.pg_vector = PgVector()
        self.llm_model = GeminiLLM()
        self.video_id = video_id

    def search_fuzzy_phrase(self, cursor, video_id, words, max_gap=0.5, similarity_threshold=0.3):

        if not words:
            return []

        # Build the query dynamically based on number of words
        tables = []
        conditions = []
        select_parts = []

        # Add table aliases and select parts
        for i in range(len(words)):
            alias = f"w{i}"
            tables.append(f"wordstimestamp {alias}")
            select_parts.append(f"{alias}.word AS word{i}, {alias}.start_time AS start{i}, {alias}.end_time AS end{i}")

        # Add basic conditions
        conditions.append(f"w0.video_id = %s")

        # Add similarity conditions for each word
        for i, word in enumerate(words):
            conditions.append(f"similarity(w{i}.word, %s) > {similarity_threshold}")

        # Add consecutive word conditions
        for i in range(len(words) - 1):
            conditions.append(f"w{i}.end_time <= w{i + 1}.start_time")  # Words must be in sequence
            conditions.append(f"w{i + 1}.start_time - w{i}.end_time <= {max_gap}")  # Max gap between words

        # Construct the full query
        query = f"""
            SELECT {', '.join(select_parts)}
            FROM {tables[0]}
        """

        # Add joins for additional words
        for i in range(1, len(words)):
            query += f"""
            JOIN {tables[i]} ON w0.video_id = w{i}.video_id
            """

        # Add WHERE clause and ordering
        query += f"""
            WHERE {' AND '.join(conditions)}
            ORDER BY w0.start_time
        """

        # Prepare parameters
        params = [video_id] + words

        cursor.execute(query, params)
        return cursor.fetchall()

    def __call__(self ,*args):
        self.state = args[0]
        messages = self.state["messages"]
        user_input = messages[-1].content
        words = [word.strip() for word in user_input.split(",")]
        response = self.search_fuzzy_phrase(self.pg_vector.cursors, self.video_id, words, max_gap=0.5, similarity_threshold=0.3)
        response = {"messages": AIMessage(content=str(response),
                                          additional_kwargs={"search_word": user_input})}
        return response


