import psycopg2
import os
from datetime import datetime
from scipy.spatial.distance import cosine


class PgVector:
    def __init__(self):
        self.conn = psycopg2.connect(
            host=os.getenv("Postgres_Host"),
            port=os.getenv("Postgres_Port"),
            database=os.getenv("Postgres_Database"),
            user=os.getenv("Postgres_User"),
            password=os.getenv("Postgres_Password")
        )
        self.cursors = self.conn.cursor()
        self.init_table()

        self.graph_similarity_threshold = 0.8


    def init_table(self):
        self.cursors.execute("CREATE EXTENSION IF NOT EXISTS vector;")

        self.cursors.execute("""CREATE TABLE IF NOT EXISTS sources (
            id SERIAL PRIMARY KEY,
            source_url TEXT UNIQUE NOT NULL,
            text TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );""")

        self.cursors.execute("""CREATE TABLE IF NOT EXISTS embeddings (
            id SERIAL PRIMARY KEY,
            video_id INTEGER REFERENCES sources(id) ON DELETE CASCADE,
            text_chunk TEXT NOT NULL,
            embedding vector(768) NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );""")

        self.cursors.execute("""CREATE TABLE IF NOT EXISTS graph_edges (
            source_id INTEGER REFERENCES embeddings(id) ON DELETE CASCADE,
            target_id INTEGER REFERENCES embeddings(id) ON DELETE CASCADE,
            similarity FLOAT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (source_id, target_id)
        );""")

        self.cursors.execute("""CREATE TABLE IF NOT EXISTS wordstimestamp (
            source TEXT PRIMARY KEY,
            text TEXT NOT NULL,
            word TEXT NOT NULL,
            start_time FLOAT NOT NULL,
            end_time FLOAT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );""")

        self.cursors.execute("""CREATE INDEX IF NOT EXISTS embeddings_vector_idx ON embeddings
                                USING hnsw (embedding vector_l2_ops) WITH (m = 16, ef_construction = 64);
                             """)
        self.cursors.execute("CREATE INDEX IF NOT EXISTS graph_edges_source_idx ON graph_edges (source_id);")
        self.cursors.execute("CREATE INDEX IF NOT EXISTS graph_edges_target_idx ON graph_edges (target_id);")

        self.conn.commit()


    def is_source_available(self, source_url: str):
        check_query = "SELECT id FROM sources WHERE source_url = %s LIMIT 1;"
        self.cursors.execute(check_query, (source_url,))
        existing = self.cursors.fetchone()
        if existing:
            return True
        else:
            return False

    def insert_source(self, source_url: str, text: str):
        # First, check if source_url already exists
        check_query = "SELECT id FROM sources WHERE source_url = %s;"
        self.cursors.execute(check_query, (source_url,))
        existing = self.cursors.fetchone()



        insert_query = """
        INSERT INTO sources (source_url, text)
        VALUES (%s, %s)
        RETURNING id;
        """
        self.cursors.execute(insert_query,(source_url, text))
        source_id = self.cursors.fetchone()[0]
        self.conn.commit()
        return source_id


    def insert_embedding(self, video_id, text_chunk, embedding):
        insert_query = """
        INSERT INTO embeddings (video_id, text_chunk, embedding)
        VALUES (%s, %s, %s)
        RETURNING id;
        """
        self.cursors.execute(insert_query, (video_id, text_chunk, embedding))
        embedding_id = self.cursors.fetchone()[0]
        self.conn.commit()
        return embedding_id

    def create_graph_connections(self, nodes):
        for id1, vec1 in nodes:
            for id2, vec2 in nodes:
                if id1 != id2:
                    similarity = 1 - cosine(vec1, vec2)
                    if similarity > self.graph_similarity_threshold:
                        insert_query = """
                        INSERT INTO graph_edges (source_id, target_id, similarity)
                        VALUES (%s, %s, %s) ON CONFLICT DO NOTHING;
                        """
                        self.cursors.execute(insert_query, (id1, id2, similarity))
        self.conn.commit()
