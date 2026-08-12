import os
import requests
from typing import Dict, Any, List
import streamlit as st
from qdrant_client import QdrantClient
from langchain_huggingface import HuggingFaceEmbeddings
from groq import Groq
from tavily import TavilyClient


# --- BEZBEDNO ČITANJE PODEŠAVANJA ---
def get_config(key: str, default: str = "") -> str:
    """Čita varijable prvo iz Streamlit Secrets, pa iz okruženja (env)."""
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
TAVILY_API_KEY = get_config("TAVILY_API_KEY", "")

# Online / Remote Hosted Ollama podešavanja
OLLAMA_URL = get_config("OLLAMA_URL", "https://tvoj-online-ollama-endpoint.com/api/chat")
OLLAMA_API_KEY = get_config("OLLAMA_API_KEY", "")
OLLAMA_MODEL = get_config("OLLAMA_MODEL", "qwen2.5:7b")


# --- KEŠIRANE KONEKCIJE ---
@st.cache_resource(show_spinner=False)
def load_embedding_model():
    return HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )


@st.cache_resource(show_spinner=False)
def get_qdrant_client():
    return QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, check_compatibility=False)


def get_groq_client() -> Groq | None:
    key = get_config("GROQ_API_KEY", "")
    return Groq(api_key=key) if key else None


# --- PRETRAGA INTERNE QDRANT BAZE ---
def retrieve_internal_context(query: str, top_k: int = 3, max_chars_per_doc: int = 3500) -> Dict[str, Any]:
    """Pretražuje Qdrant i vraća tekst skraćen na bezbednu dužinu."""
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
    max_score = 0.0

    for res in results:
        if res.score > max_score:
            max_score = res.score

        payload = res.payload
        doc_name = payload.get("document_name", "Dokument")
        content = payload.get("markdown_content", "")
        
        # Skraćivanje predugačkog teksta radi izbegavanja token limita
        content_snippet = content[:max_chars_per_doc] + ("\n...[skraćeno]" if len(content) > max_chars_per_doc else "")
        
        employees = payload.get("detected_employees_in_doc", [])
        emp_str = ", ".join([e['employee_name'] for e in employees]) if employees else "Nema"
        
        csvs = payload.get("csv_tables_attached", [])
        csv_str = ", ".join(csvs) if csvs else "Nema"

        block = f"📄 INTERNI DOKUMENT: {doc_name}\n"
        block += f"👤 PREPOZNATI ZAPOSLENI: {emp_str}\n"
        block += f"📊 TABELE: {csv_str}\n"
        block += f"📝 TEKST:\n{content_snippet}\n"
        
        formatted_context.append(block)
        sources.append(doc_name)

    return {
        "context_str": "\n" + "="*20 + "\n" + "\n=".join(formatted_context),
        "sources": list(set(sources)),
        "max_score": max_score
    }


# --- WEB PRETRAGA (TAVILY) ---
def search_web_tavily(query: str) -> Dict[str, Any]:
    """Pretražuje internet preko Tavily API-ja ako internih podataka nema."""
    api_key = get_config("TAVILY_API_KEY", "")
    if not api_key:
        return {"web_context": "", "web_sources": []}

    try:
        tavily = TavilyClient(api_key=api_key)
        response = tavily.search(query=query, search_depth="basic", max_results=3)
        
        web_blocks = []
        web_sources = []
        for result in response.get("results", []):
            title = result.get("title", "Web Izvor")
            url = result.get("url", "")
            snippet = result.get("content", "")
            
            web_blocks.append(f"🌐 WEB IZVOR: {title} ({url})\n📝 SADRŽAJ:\n{snippet}\n")
            web_sources.append(url)

        return {
            "web_context": "\n".join(web_blocks),
            "web_sources": web_sources
        }
    except Exception as e:
        print(f"⚠️ Tavily greška: {e}")
        return {"web_context": "", "web_sources": []}


# --- LLM INVOCATIONS ---
def query_groq(system_prompt: str, user_query: str) -> str:
    client = get_groq_client()
    if not client:
        raise ValueError("GROQ_API_KEY nije definisan.")
        
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


def query_ollama_online(system_prompt: str, user_query: str) -> str:
    """Poziva eksterni / hosted Online Ollama API sa API ključem."""
    headers = {"Content-Type": "application/json"}
    if OLLAMA_API_KEY:
        headers["Authorization"] = f"Bearer {OLLAMA_API_KEY}"

    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query}
        ],
        "stream": False
    }
    
    res = requests.post(OLLAMA_URL, json=payload, headers=headers, timeout=60)
    res.raise_for_status()
    return res.json()["message"]["content"]


# --- GLAVNI RAG FLOW ---
def ask_birochat(user_query: str) -> Dict[str, Any]:
    # 1. Pretraga interne baze
    internal_res = retrieve_internal_context(user_query)
    
    used_web_search = False
    web_sources = []
    combined_context = ""

    # Ako interni rezultati imaju nizak score (< 0.40), aktivira se Tavily Web Search
    if internal_res["max_score"] < 0.40 or not internal_res["sources"]:
        used_web_search = True
        web_res = search_web_tavily(user_query)
        web_sources = web_res["web_sources"]
        
        combined_context = (
            "⚠️ Napomena: Podaci nisu pronađeni u internoj bazi. Korišćena je pretraga sa interneta.\n\n"
            f"{web_res['web_context']}"
        )
    else:
        combined_context = f"INTERNI PODACI:\n{internal_res['context_str']}"

    system_prompt = (
        "Ti si BiroChat AI asistent. Odgovaraj precizno, tačno i profesionalno na srpskom jeziku.\n"
        "Za formulisanje odgovora primarno koristi priložene podatke.\n"
        "Ako su priloženi interni podaci, navedi i detalje o zaposlenima ili tabelama ako ih ima.\n\n"
        f"PRILOŽENI PODACI:\n{combined_context}"
    )

    used_provider = "Groq (LLaMA-3.3-70B)"
    
    # Executing LLM call
    try:
        answer = query_groq(system_prompt, user_query)
    except Exception as groq_err:
        print(f"⚠️ Groq greška ({groq_err}). Preusmeravam na Online Ollama...")
        try:
            used_provider = f"Online Ollama ({OLLAMA_MODEL})"
            answer = query_ollama_online(system_prompt, user_query)
        except Exception as fallback_err:
            used_provider = "Greška"
            answer = f"❌ Greška pri komunikaciji sa modelima:\n- Groq: `{groq_err}`\n- Ollama: `{fallback_err}`"

    return {
        "answer": answer,
        "sources": internal_res["sources"],
        "web_sources": web_sources,
        "used_web": used_web_search,
        "provider": used_provider
    }