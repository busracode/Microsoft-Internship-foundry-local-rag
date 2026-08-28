"""
Sets up the database by creating and populating it with document embeddings.
Usage: python main.py
(Run this once; afterwards, launch the UI with 'streamlit run app.py'.)
"""

import sqlite3
import numpy as np
from foundry_local_sdk import Configuration, FoundryLocalManager


def vector_to_blob(vector):
    """Converts an embedding vector into a byte blob for SQLite storage."""
    return np.array(vector, dtype=np.float32).tobytes()


# Knowledge base: the source documents the chatbot will answer from
documents = [
    "Foundry Local is a Microsoft tool that runs AI models directly on-device, without a cloud connection.",
    "RAG answers a question by first retrieving relevant documents, then augmenting the prompt with that information, and finally generating an answer with the model.",
    "Embedding is the process of converting text into a numerical vector; semantically similar texts end up close together in vector space.",
    "Cosine similarity measures the angular similarity between two vectors, producing a result between -1 and 1.",
    "SQLite is a lightweight, serverless database that runs from a single file.",
    "Chunking is the process of splitting a long document into smaller pieces, which helps embeddings produce more accurate results.",
    "Prompt engineering is the process of designing system and user prompts to achieve the desired behavior from a language model.",
    "Foundry Local's catalog includes both small, fast models and large, powerful ones; the choice depends on the speed-quality tradeoff.",
]


def setup_database():
    """Initializes Foundry Local, embeds the documents, and stores them in SQLite."""
    config = Configuration(app_name="foundry-rag-project")
    FoundryLocalManager.initialize(config)
    manager = FoundryLocalManager.instance

    embed_model = manager.catalog.get_model("qwen3-embedding-0.6b")
    print("Downloading embedding model...")
    embed_model.download(lambda p: print(f"\rDownloading: {p:.1f}%", end="", flush=True))
    print()
    embed_model.load()
    embed_client = embed_model.get_embedding_client()

    conn = sqlite3.connect("rag_database.db")
    cursor = conn.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS documents (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        text TEXT NOT NULL,
        embedding BLOB NOT NULL
    )
    """)

    cursor.execute("SELECT COUNT(*) FROM documents")
    existing_count = cursor.fetchone()[0]

    if existing_count == 0:
        print(f"Processing {len(documents)} documents...")
        response = embed_client.generate_embeddings(documents)
        embeddings = [item.embedding for item in response.data]
        for text, emb in zip(documents, embeddings):
            cursor.execute(
                "INSERT INTO documents (text, embedding) VALUES (?, ?)",
                (text, vector_to_blob(emb))
            )
        conn.commit()
        print(f"{len(documents)} documents saved to the database.")
    else:
        print(f"Database already populated ({existing_count} records), skipping insert.")

    conn.close()


if __name__ == "__main__":
    setup_database()
    print("\nSetup complete. Now run 'streamlit run app.py' to launch the UI.")