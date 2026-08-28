"""
Streamlit web interface for the Foundry Local RAG Assistant.
Run with: streamlit run app.py
"""

import streamlit as st
from rag_core import RagSystem

st.set_page_config(page_title="Foundry Local RAG Assistant", page_icon="🤖")

st.title("🤖 Foundry Local RAG Assistant")
st.caption("A fully offline, on-device question-answering system")

# Load models only once and keep them in session_state.
# Without this, Streamlit would reload both models on every user interaction
# (e.g. every time a question is asked), which would be extremely slow.
if "rag_system" not in st.session_state:
    with st.spinner("Loading models, this may take a moment the first time..."):
        st.session_state.rag_system = RagSystem()
    st.success("Models ready!")

# Chat history is also kept in session_state so it survives re-runs
# (Streamlit re-executes this whole script on every interaction).
if "messages" not in st.session_state:
    st.session_state.messages = []

# Re-render the full conversation history on every run.
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# Handle a new question typed by the user.
if question := st.chat_input("Ask a question..."):
    # Show and store the user's message.
    with st.chat_message("user"):
        st.write(question)
    st.session_state.messages.append({"role": "user", "content": question})

    # Run the full RAG pipeline (retrieve -> augment -> generate) and display the answer.
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer, top_chunks = st.session_state.rag_system.answer_query(question)
        st.write(answer)

        # Show which knowledge base entries were used, for transparency.
        with st.expander("Sources used"):
            for text, score in top_chunks:
                st.write(f"**(score: {score:.3f})** {text}")

    st.session_state.messages.append({"role": "assistant", "content": answer})