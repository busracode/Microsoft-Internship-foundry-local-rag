import sqlite3
import numpy as np
from foundry_local_sdk import Configuration, FoundryLocalManager


def vector_to_blob(vector):
    return np.array(vector, dtype=np.float32).tobytes()


def blob_to_vector(blob):
    return np.frombuffer(blob, dtype=np.float32)


def cosine_similarity(v1, v2):
    v1, v2 = np.array(v1), np.array(v2)
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))


def get_top_chunks(query, embed_client, top_n=3):
    response = embed_client.generate_embedding(query)
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


def answer_query(query, embed_client, chat_client, top_n=3):
    # 1. Retrieve
    top_chunks = get_top_chunks(query, embed_client, top_n=top_n)
    context = "\n".join([f"- {text}" for text, score in top_chunks])

    # 2. Augment
    system_prompt = (
        "Sen sadece verilen bağlamı kullanarak cevap veren bir asistansın. "
        "Bağlamda olmayan bir şey soruluyorsa 'Bu bilgi elimde yok' de.\n\n"
        f"Bağlam:\n{context}"
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": query},
    ]

    # 3. Generate
    response = chat_client.complete_chat(messages)
    answer = response.choices[0].message.content

    return answer, top_chunks


if __name__ == "__main__":
    # Foundry Local'i başlat
    config = Configuration(app_name="foundry-rag-project")
    FoundryLocalManager.initialize(config)
    manager = FoundryLocalManager.instance

    # Embedding modeli (veritabanı zaten dolu, sadece retrieval için lazım)
    embed_model = manager.catalog.get_model("qwen3-embedding-0.6b")
    embed_model.download(lambda p: print(f"\rEmbedding modeli: %{p:.1f}", end="", flush=True))
    print()
    embed_model.load()
    embed_client = embed_model.get_embedding_client()

    # Chat modeli
    chat_model = manager.catalog.get_model("phi-4-mini")
    chat_model.download(lambda p: print(f"\rChat modeli: %{p:.1f}", end="", flush=True))
    print()
    chat_model.load()
    chat_client = chat_model.get_chat_client()

    print("\nModeller hazır.\n")

    # Test soruları
    test_questions = [
        "Uzun bir belgeyi nasıl parçalara bölerim?",
        "Foundry Local ne işe yarar?",
        "Bugün hava nasıl?",  # bağlam dışı, "bilmiyorum" demeli
    ]

    for q in test_questions:
        print(f"Soru: {q}")
        answer, chunks = answer_query(q, embed_client, chat_client)
        print(f"Cevap: {answer}")
        print(f"(Kullanılan bağlam: {[c[0][:40] + '...' for c in chunks]})")
        print("-" * 60)