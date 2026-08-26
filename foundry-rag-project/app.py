import streamlit as st
from rag_core import RagSystem

st.set_page_config(page_title="Foundry Local RAG Asistanı", page_icon="🤖")

st.title("🤖 Foundry Local RAG Asistanı")
st.caption("Tamamen internetsiz, cihaz üzerinde çalışan soru-cevap sistemi")

# Modelleri sadece BİR KEZ yükle (Streamlit her etkileşimde sayfayı yeniden çalıştırır,
# bu yüzden session_state kullanarak modelleri hafızada tutuyoruz)
if "rag_system" not in st.session_state:
    with st.spinner("Modeller yükleniyor, bu ilk seferde biraz sürebilir..."):
        st.session_state.rag_system = RagSystem()
    st.success("Modeller hazır!")

# Sohbet geçmişini tutmak için
if "messages" not in st.session_state:
    st.session_state.messages = []

# Geçmiş mesajları ekrana yazdır
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# Kullanıcıdan yeni soru al
if question := st.chat_input("Sorunuzu yazın..."):
    # Kullanıcı mesajını göster ve kaydet
    with st.chat_message("user"):
        st.write(question)
    st.session_state.messages.append({"role": "user", "content": question})

    # Cevabı üret
    with st.chat_message("assistant"):
        with st.spinner("Düşünüyorum..."):
            answer, top_chunks = st.session_state.rag_system.answer_query(question)
        st.write(answer)

        # Kullanılan kaynakları göster (şeffaflık için)
        with st.expander("Kullanılan kaynaklar"):
            for text, score in top_chunks:
                st.write(f"**(skor: {score:.3f})** {text}")

    st.session_state.messages.append({"role": "assistant", "content": answer})
