import os
import requests
from typing import Dict, Any, List
import streamlit as st
from qdrant_client import QdrantClient
from langchain_huggingface import HuggingFaceEmbeddings
from groq import Groq


# --- POMOĆNA FUNKCIJA ZA BEZBEDNO DOBIJANJE KLJUČEVA ---
def get_config(key: str, default: str = "") -> str:
    """Čita ključ prvo iz Streamlit Secrets (za Cloud deployment), pa iz os.getenv (za lokalno)."""
    try:
        if key in st.secrets:
            return str(st.secrets[key])
    except Exception:
        pass
    return os.getenv(key, default)


# --- KONFIGURACIJA ---
QDRANT_URL = get_config("QDRANT_URL", "https://09ffa5ef-2765-45c8-bfcf-29bc6bf90f08.eu-west-2-0.aws.cloud.qdrant.io")
QDRANT_API_KEY = get_config("QDRANT_API_KEY", "")
GROQ_API_KEY = get_config("GROQ_API_KEY", "")

OLLAMA_URL = get_config("OLLAMA_URL", "http://localhost:11434/api/chat")
OLLAMA_MODEL = get_config("OLLAMA_MODEL", "qwen2.5:7b")


# --- KEŠIRANE RESURSNE FUNKCIJE (Sprečavaju usporenje u Streamlit-u) ---
@st.cache_resource(show_spinner=False)
def load_embedding_model():
    """Učitava BGE-M3 model jednom i drži ga u memoriji."""
    return HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )


@st.cache_resource(show_spinner=False)
def get_qdrant_client():
    """Inicijalizuje Qdrant klijent konekciju."""
    return QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, check_compatibility=False)


def get_groq_client() -> Groq | None:
    """Dinamički vraća Groq klijent ako je API ključ dostupan."""
    key = get_config("GROQ_API_KEY", "")
    if key:
        return Groq(api_key=key)
    return None


# --- RETRIEVAL LOGIKA ---
def retrieve_context(query: str, top_k: int = 3) -> Dict[str, Any]:
    """Pretražuje Qdrant i priprema strukturisani kontekst iz pronađenih dokumenata."""
    embed_model = load_embedding_model()
    qdrant_client = get_qdrant_client()

    query_vector = embed_model.embed_query(query)
    
    response = qdrant_client.query_points(
        collection_name="multimodal_documents",
        query=query_vector,
        limit=top_k
    )
    results = response.points

    formatted_context = []
    sources = []

    for res in results:
        payload = res.payload
        doc_name = payload.get("document_name", "Nenaveden dokument")
        content = payload.get("markdown_content", "")
        
        employees = payload.get("detected_employees_in_doc", [])
        emp_str = ", ".join([e['employee_name'] for e in employees]) if employees else "Nema detektovanih lica"
        
        csvs = payload.get("csv_tables_attached", [])
        csv_str = ", ".join(csvs) if csvs else "Nema priloženih tabela"

        block = f"📄 DOKUMENT: {doc_name}\n"
        block += f"👤 PREPOZNATI ZAPOSLENI: {emp_str}\n"
        block += f"📊 ATTACHED TABELE: {csv_str}\n"
        block += f"📝 TEKST DOKUMENTA:\n{content}\n"
        
        formatted_context.append(block)
        sources.append(doc_name)

    return {
        "full_context": "\n" + "="*20 + "\n" + "\n=".join(formatted_context),
        "sources": list(set(sources))
    }


# --- LLM POZIVI ---
def query_groq(system_prompt: str, user_query: str) -> str:
    """Poziva Groq API sa LLaMA-3.3-70B modelom."""
    client = get_groq_client()
    if not client:
        raise ValueError("GROQ_API_KEY nije definisan u okruženju niti u Streamlit secrets-u.")
        
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query}
        ],
        temperature=0.2,
        max_tokens=1500
    )
    return response.choices[0].message.content


def query_ollama_fallback(system_prompt: str, user_query: str) -> str:
    """Poziva lokalni Ollama API u slučaju da je Groq nedostupan."""
    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query}
        ],
        "stream": False
    }
    res = requests.post(OLLAMA_URL, json=payload, timeout=45)
    res.raise_for_status()
    return res.json()["message"]["content"]


# --- GLAVNA RAG FUNKCIJA ---
def ask_birochat(user_query: str) -> Dict[str, Any]:
    """Glavna funkcija: pretražuje bazu, formira prompt i generiše odgovor sa fallback mehanizmom."""
    retrieved = retrieve_context(user_query)
    
    system_prompt = (
        "Ti si BiroChat AI asistent. Odgovaraj precizno, tačno i profesionalno na srpskom jeziku.\n"
        "Koristi ISKLJUČIVO dole priloženi kontekst iz interne baze za formulisanje odgovora.\n"
        "Ako u kontekstu postoje informacije o prepoznatim zaposlenima na slikama ili priloženim CSV tabelama, obavezno ih navedi u odgovoru.\n"
        "Ako priloženi kontekst ne sadrži odgovor na pitanje, izričito navedi da te informacije ne postoje u internoj bazi.\n\n"
        f"KONTEKST IZ INTERNE BAZE:\n{retrieved['full_context']}"
    )

    used_provider = "Groq (LLaMA-3.3-70B)"
    
    # 1. Pokušaj izvršavanja preko Groq API-ja
    try:
        answer = query_groq(system_prompt, user_query)
    except Exception as groq_err:
        print(f"⚠️ Groq API greška ({groq_err}). Pokušavam lokalni Ollama fallback...")
        # 2. Fallback na Ollama-u ako Groq otkaže (radi lokalno)
        try:
            used_provider = f"Lokalni Ollama ({OLLAMA_MODEL})"
            answer = query_ollama_fallback(system_prompt, user_query)
        except Exception as fallback_err:
            used_provider = "Nijedan (Greška)"
            answer = (
                f"❌ Došlo je do greške pri komunikaciji sa AI modelima.\n"
                f"- Groq greška: `{groq_err}`\n"
                f"- Ollama fallback greška: `{fallback_err}`"
            )

    return {
        "answer": answer,
        "sources": retrieved["sources"],
        "provider": used_provider
    }


if __name__ == "__main__":
    # Test pokretanje iz konzole
    print("🧪 Testiram RAG Engine iz konzole...")
    test_res = ask_birochat("Pronađi sve analitičke kartice i izvod za Srbijašume.")
    print("\n--- ODGOVOR ---")
    print(test_res["answer"])
    print("\n--- METAPODACI ---")
    print(f"Provider: {test_res['provider']}")
    print(f"Izvori: {test_res['sources']}")