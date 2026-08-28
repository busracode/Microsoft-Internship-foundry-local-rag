# Foundry Local RAG Assistant

A fully offline, on-device question-answering assistant built with [Microsoft Foundry Local](https://learn.microsoft.com/en-us/azure/foundry-local/). It uses Retrieval-Augmented Generation (RAG) to answer questions based on a local knowledge base — no cloud connection, no API keys, no internet required after setup.

## 🎥 Demo Video

[![Watch Demo](https://img.shields.io/badge/YouTube-Watch%20Demo-red?logo=youtube&logoColor=white&style=for-the-badge)](https://www.youtube.com/watch?v=DnA4s5s0tMo)

> Watch the full walkthrough — architecture explanation, live demo, and lessons learned.

## How It Works

1. **Retrieve** — The user's question is converted into a vector (embedding) and compared against a knowledge base of pre-embedded document chunks using cosine similarity. The most relevant chunks are selected.
2. **Augment** — The retrieved chunks are inserted into the system prompt as context for the language model.
3. **Generate** — A local language model generates an answer based strictly on the provided context. If the context doesn't contain enough information, the model responds with "I don't have this information."

All processing — embedding generation and text generation — runs entirely on-device via Foundry Local.

## Tech Stack

- **[Foundry Local](https://learn.microsoft.com/en-us/azure/foundry-local/)** — on-device model runtime (no cloud/GPU server required)
- **Embedding model:** `qwen3-embedding-0.6b`
- **Chat model:** `phi-4-mini`
- **SQLite** — lightweight local storage for document text and embeddings
- **Streamlit** — web-based chat interface
- **NumPy** — vector math (cosine similarity)

## Project Structure

```
foundry-rag-project/
├── main.py            # One-time setup script: embeds documents and populates the database
├── rag_core.py         # Core RAG logic (RagSystem class): retrieval, prompting, generation
├── app.py              # Streamlit web interface
├── requirements.txt    # Python dependencies
└── rag_database.db     # SQLite database (created after running main.py)
```

## Setup & Installation

**Prerequisites:** Python 3.11 or 3.12, [Foundry Local](https://learn.microsoft.com/en-us/azure/foundry-local/) support on your machine.

1. Clone the repository and navigate into it:
```bash
   git clone https://github.com/busracode/Microsoft-Internship-foundry-local-rag.git
   cd foundry-rag-project
```

2. Create and activate a virtual environment:
```bash
   python -m venv venv
   # Windows
   venv\Scripts\Activate.ps1
   # macOS/Linux
   source venv/bin/activate
```

3. Install dependencies:
```bash
   pip install -r requirements.txt
```

4. Set up the database (downloads the embedding model and embeds the knowledge base — first run only):
```bash
   python main.py
```

5. Launch the app:
```bash
   streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

## Usage

Type a question in the chat box. The assistant will:
- Find the most relevant pieces of information from its knowledge base
- Generate an answer grounded in that information
- Show the sources it used under "Sources used" for transparency

## Known Limitations

- **Response time:** ~10–13 seconds per query due to the size of the local chat model (`phi-4-mini`) running on CPU. This could be improved by using a smaller/quantized model, at some cost to answer quality.
- **Small knowledge base:** The current demo uses a handful of short documents about RAG concepts. Swap in your own documents in `main.py` to build a knowledge base for a different topic.
- **No incremental updates:** Adding new documents currently requires re-running the ingestion logic; there's no "add a document" UI yet.

## Possible Future Improvements

- Support uploading custom documents (PDF/TXT) directly from the UI
- Add proper text chunking for longer documents (currently each document is treated as a single chunk)
- Experiment with smaller/quantized models to reduce response time
- Add automated tests for the retrieval and generation pipeline

## Acknowledgments

Built as part of a summer internship project exploring on-device RAG applications with Microsoft Foundry Local.
