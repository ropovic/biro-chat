import os
import streamlit as st
from rag_engine import ask_birochat

st.set_page_config(page_title="BiroChat AI", page_icon="🤖", layout="wide")

st.sidebar.title("🤖 BiroChat Status")
st.sidebar.markdown("---")

groq_set = bool(os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY"))
st.sidebar.success("✅ Primarni LLM: Groq" if groq_set else "⚠️ Groq nije podešen")
st.sidebar.info("🌐 Web Fallback: Tavily Search")
st.sidebar.info("🎯 Vector Engine: Qdrant Cloud (BGE-M3)")

st.sidebar.markdown("---")
if st.sidebar.button("🧹 Očisti chat"):
    st.session_state.messages = []
    st.rerun()

st.title("📂 BiroChat — Hybrid RAG Asistent")
st.caption("Pretražuje internu bazu Biroa, a po potrebi vrši pretragu interneta u realnom vremenu.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and "provider" in message:
            sources = ", ".join(message.get('sources', [])) or "Nema internih"
            web_info = f" | 🌐 Web izvori: {len(message.get('web_sources', []))}" if message.get('used_web') else ""
            st.caption(f"⚙️ **Model:** {message['provider']} | 📄 **Interni izvori:** `{sources}`{web_info}")

if user_query := st.chat_input("Postavite pitanje..."):
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        with st.spinner("Pretražujem bazu i internet..."):
            res = ask_birochat(user_query)
            
            st.markdown(res["answer"])
            
            sources = ", ".join(res['sources']) if res['sources'] else "Nema internih"
            web_info = f" | 🌐 Korišćena pretraga interneta ({len(res['web_sources'])} izvora)" if res['used_web'] else ""
            st.caption(f"⚙️ **Model:** {res['provider']} | 📄 **Interni izvori:** `{sources}`{web_info}")
            
            st.session_state.messages.append({
                "role": "assistant",
                "content": res["answer"],
                "provider": res["provider"],
                "sources": res["sources"],
                "web_sources": res["web_sources"],
                "used_web": res["used_web"]
            })