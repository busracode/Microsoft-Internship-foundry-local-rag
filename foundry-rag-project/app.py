import streamlit as st
from rag_core import RagSystem

st.set_page_config(page_title="Foundry Local RAG Assistant", page_icon="🤖")

st.title("🤖 Foundry Local RAG Assistant")
st.caption("A fully offline, on-device question-answering system")

if "rag_system" not in st.session_state:
    with st.spinner("Loading models, this may take a moment the first time..."):
        st.session_state.rag_system = RagSystem()
    st.success("Models ready!")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

if question := st.chat_input("Ask a question..."):
    with st.chat_message("user"):
        st.write(question)
    st.session_state.messages.append({"role": "user", "content": question})

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer, top_chunks = st.session_state.rag_system.answer_query(question)
        st.write(answer)

        with st.expander("Sources used"):
            for text, score in top_chunks:
                st.write(f"**(score: {score:.3f})** {text}")

    st.session_state.messages.append({"role": "assistant", "content": answer})