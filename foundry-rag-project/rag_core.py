import sqlite3
import time
import numpy as np
from foundry_local_sdk import Configuration, FoundryLocalManager


def vector_to_blob(vector):
    return np.array(vector, dtype=np.float32).tobytes()


def blob_to_vector(blob):
    return np.frombuffer(blob, dtype=np.float32)


def cosine_similarity(v1, v2):
    v1, v2 = np.array(v1), np.array(v2)
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))


class RagSystem:
    """Wraps the embedding model, chat model, and database access for the RAG pipeline."""

    def __init__(self):
        config = Configuration(app_name="foundry-rag-project")
        FoundryLocalManager.initialize(config)
        manager = FoundryLocalManager.instance

        self.embed_model = manager.catalog.get_model("qwen3-embedding-0.6b")
        self.embed_model.download()
        self.embed_model.load()
        self.embed_client = self.embed_model.get_embedding_client()

        self.chat_model = manager.catalog.get_model("phi-4-mini")
        self.chat_model.download()
        self.chat_model.load()
        self.chat_client = self.chat_model.get_chat_client()

    def get_top_chunks(self, query, top_n=3):
        """Returns the top_n most relevant chunks for a given query, as (text, score) pairs."""
        response = self.embed_client.generate_embedding(query)
        query_embedding = response.data[0].embedding

        conn = sqlite3.connect("rag_database.db")
        cursor = conn.cursor()
        cursor.execute("SELECT text, embedding FROM documents")
        rows = cursor.fetchall()
        conn.close()

        scored = []
        for text, blob in rows:
            chunk_embedding = blob_to_vector(blob)
            score = cosine_similarity(query_embedding, chunk_embedding)
            scored.append((text, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_n]

    def answer_query(self, query, top_n=3):
        """Runs the full RAG pipeline: retrieve relevant chunks, augment the prompt, generate an answer."""
        start_time = time.time()

        top_chunks = self.get_top_chunks(query, top_n=top_n)
        context = "\n".join([f"- {text}" for text, score in top_chunks])

        system_prompt = (
            "You are an assistant that answers questions using only the context provided below.\n"
            "RULES:\n"
            "1. Base your answer strictly on the context below.\n"
            "2. If the context does not contain enough information to answer, reply ONLY with "
            "'I don't have this information.' and nothing else.\n"
            "3. If the context does contain relevant information, give a complete answer using those details.\n\n"
            f"Context:\n{context}"
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query},
        ]

        response = self.chat_client.complete_chat(messages)
        answer = response.choices[0].message.content

        elapsed = time.time() - start_time
        print(f"[Response time: {elapsed:.2f}s]")

        return answer, top_chunks