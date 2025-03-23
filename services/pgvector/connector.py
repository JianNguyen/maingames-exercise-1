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
        self.cursors.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm;")

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
            id SERIAL PRIMARY KEY,
            video_id INTEGER REFERENCES sources(id) ON DELETE CASCADE,
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
        self.cursors.execute("""
                            CREATE INDEX IF NOT EXISTS idx_wordstimestamp_trgm 
                            ON wordstimestamp USING GIN (word gin_trgm_ops);
                        """)
        self.conn.commit()


    def is_source_available(self, source_url: str):
        check_query = "SELECT id FROM sources WHERE source_url = %s LIMIT 1;"
        self.cursors.execute(check_query, (source_url,))
        existing = self.cursors.fetchone()
        if existing:
            return existing[0]
        else:
            return False

    def insert_to_sources_tb(self, source_url: str, text: str):
        insert_query = """
        INSERT INTO sources (source_url, text)
        VALUES (%s, %s)
        RETURNING id;
        """
        self.cursors.execute(insert_query,(source_url, text))
        source_id = self.cursors.fetchone()[0]
        self.conn.commit()
        return source_id


    def insert_embedding_to_embeddings_tb(self, video_id, text_chunk, embedding):
        insert_query = """
        INSERT INTO embeddings (video_id, text_chunk, embedding)
        VALUES (%s, %s, %s)
        RETURNING id;
        """
        self.cursors.execute(insert_query, (video_id, text_chunk, embedding))
        embedding_id = self.cursors.fetchone()[0]
        self.conn.commit()
        return embedding_id

    def insert_multiple_embeddings_to_embeddings_tb(self, video_id, embeds):
        graph_nodes = []
        for embed in embeds:
            node = []
            embedding_id = self.insert_embedding_to_embeddings_tb(video_id, embed["text_chunk"], embed["embedding"])
            node.append(embedding_id)
            node.append(embed["embedding"])
            graph_nodes.append(node)

        return graph_nodes

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

    def insert_words_timestamp_to_wordstimestamp_tb(self, video_id, els):
        for el in els:
            insert_query = """INSERT INTO wordstimestamp (video_id, word, start_time, end_time)
                              VALUES (%s, %s, %s, %s);
                           """
            self.cursors.execute(insert_query, (video_id, el["word"], el["start"], el["end"]))
        self.conn.commit()

    def search_vector(self, video_id, query_vector, distance_threshold=0.45, weight_threshold=0.6, limit=3):
        self.cursors.execute("""
            SELECT * 
            FROM (
                SELECT id, text_chunk, (1 - (embedding <=> %s::vector)) as distance 
                FROM embeddings
                WHERE video_id = %s
            )
            WHERE distance > %s
            ORDER BY distance LIMIT %s;
        """, (query_vector, video_id, distance_threshold, limit, ))
        top_results = self.cursors.fetchall()
        if not top_results:
            return [],[]

        seed_ids = [row[0] for row in top_results]

        self.cursors.execute("""
            SELECT DISTINCT target_id FROM graph_edges
            WHERE source_id IN %s;
        """, (tuple(seed_ids),))
        expanded_ids = {row[0] for row in self.cursors.fetchall()}
        if not expanded_ids:
            return top_results, []

        self.cursors.execute("""
            SELECT text_chunk FROM embeddings
            WHERE id IN %s
            AND similarity >= %s;
        """, (tuple(expanded_ids), weight_threshold))
        additional_results = self.cursors.fetchall()

        return top_results, additional_results