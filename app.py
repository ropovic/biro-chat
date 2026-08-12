import os
import streamlit as st
from rag_engine import ask_birochat

# --- PODEŠAVANJE STRANICE ---
st.set_page_config(
    page_title="BiroChat AI",
    page_icon="🤖",
    layout="wide"
)

# --- SIDEBAR (STATUS I KONFIGURACIJA) ---
st.sidebar.title("🤖 BiroChat Status")
st.sidebar.markdown("---")

groq_key_set = bool(os.getenv("GROQ_API_KEY"))
if groq_key_set:
    st.sidebar.success("✅ Primarni LLM: Groq (LLaMA-3.3-70B)")
else:
    st.sidebar.warning("⚠️ Groq API ključ nije u .env — direktno se koristi Ollama")

st.sidebar.info("🛡️ Fallback LLM: Lokalni Ollama")
st.sidebar.info("🎯 Vector Engine: Qdrant Cloud (1024D BGE-M3)")

st.sidebar.markdown("---")
if st.sidebar.button("🧹 Očisti istoriju chata"):
    st.session_state.messages = []
    st.rerun()

# --- GLAVNI INTERFEJS ---
st.title("📂 BiroChat — Multimodalni RAG Asistent")
st.caption("Inteligenta pretraga i analiza internih dokumenata, finansijskih izvoda, tabela i kartona zaposlenih")

# Inicijalizacija istorije poruka u session state-u
if "messages" not in st.session_state:
    st.session_state.messages = []

# Prikaz dosadašnjih poruka iz istorije
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and "provider" in message:
            sources_str = ", ".join(message['sources']) if message['sources'] else "Nema direktnih izvora"
            st.caption(f"⚙️ **Izvršio:** {message['provider']} | 📄 **Izvori:** `{sources_str}`")

# Unos novog pitanja
if user_query := st.chat_input("Postavite pitanje o dokumentima (npr. 'Koji izvodi postoje za Srbijašume?')..."):
    # Prikaz korisničkog pitanja
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    # Obrada i generisanje odgovora
    with st.chat_message("assistant"):
        with st.spinner("🔍 Pretražujem bazu i analiziram sadržaj..."):
            res = ask_birochat(user_query)
            
            st.markdown(res["answer"])
            
            sources_str = ", ".join(res['sources']) if res['sources'] else "Nema"
            st.caption(f"⚙️ **Izvršio:** {res['provider']} | 📄 **Izvori:** `{sources_str}`")
            
            # Pamćenje odgovora u istoriji
            st.session_state.messages.append({
                "role": "assistant",
                "content": res["answer"],
                "provider": res["provider"],
                "sources": res["sources"]
            })