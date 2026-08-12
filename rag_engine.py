import os
import requests
from typing import Dict, Any, List, Tuple
import streamlit as st
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from groq import Groq
from tavily import TavilyClient


# --- BEZBEDNO ČITANJE PODEŠAVANJA (BEZ CONFIG.PY) ---
def get_config(key: str, default: str = "") -> str:
    """Čita varijable prvo iz Streamlit Secrets / Hugging Face Secrets, pa iz okruženja (env)."""
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
GEMINI_API_KEY = get_config("GEMINI_API_KEY", "")
COLLECTION_NAME = get_config("COLLECTION_NAME", "biro_documents")


# --- KEŠIRANI MODELI I KLIJENTI ---
@st.cache_resource(show_spinner=False)
def load_embedding_model():
    return SentenceTransformer("BAAI/bge-m3")


@st.cache_resource(show_spinner=False)
def get_qdrant_client():
    return QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, check_compatibility=False)


def get_groq_client() -> Groq | None:
    key = get_config("GROQ_API_KEY", "")
    return Groq(api_key=key) if key else None


# --- PRETRAGA INTERNE QDRANT BAZE ---
def retrieve_internal_context(query: str, top_k: int = 6, max_chars_per_doc: int = 2500) -> Dict[str, Any]:
    embed_model = load_embedding_model()
    qdrant_client = get_qdrant_client()

    query_vector = embed_model.encode(query).tolist()
    
    try:
        response = qdrant_client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            limit=top_k
        )
        results = response.points
    except Exception as e:
        print(f"⚠️ Qdrant pretraga greška: {e}")
        results = []

    formatted_context = []
    sources = []
    matched_employees = []
    max_score = 0.0

    for res in results:
        if res.score > max_score:
            max_score = res.score

        payload = res.payload or {}
        doc_type = payload.get("doc_type", "document")
        
        # Ako je profil zaposlenog ili logo
        if doc_type in ["employee", "employee_profile", "logo"]:
            matched_employees.append(payload)
            name = payload.get("name", "Nepoznato")
            role = payload.get("role", "Zaposleni")
            desc = payload.get("description", "")
            
            block = f"👤 PROFIL / LICE: {name}\n"
            block += f"💼 FUNKCIJA / ROLA: {role}\n"
            block += f"📝 DETALJAN OPIS: {desc}\n"
            formatted_context.append(block)
            sources.append(f"Profil: {name}")
            continue

        # Standardni interni dokumenti
        doc_name = payload.get("document_name", payload.get("name", "Dokument"))
        content = payload.get("markdown_content", payload.get("text", ""))
        
        if len(content) > max_chars_per_doc:
            content_snippet = content[:max_chars_per_doc] + "\n\n... [skraćeno zbog dužine]"
        else:
            content_snippet = content

        block = f"📄 DOKUMENT: {doc_name}\n📝 SADRŽAJ:\n{content_snippet}\n"
        formatted_context.append(block)
        sources.append(doc_name)

    return {
        "context_str": "\n" + "="*20 + "\n" + "\n=".join(formatted_context),
        "sources": list(set(sources)),
        "max_score": max_score,
        "matched_employees": matched_employees
    }


# --- WEB PRETRAGA (TAVILY FALLBACK) ---
def search_web_tavily(query: str) -> Dict[str, Any]:
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


# --- LLM POZIVI ---
def query_groq(system_prompt: str, user_query: str) -> str:
    client = get_groq_client()
    if not client:
        raise ValueError("GROQ_API_KEY nije definisan u Secrets.")
        
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query}
        ],
        temperature=0.1,
        max_tokens=1500
    )
    return response.choices[0].message.content


def query_gemini_online(system_prompt: str, user_query: str) -> str:
    gemini_key = get_config("GEMINI_API_KEY", "")
    if not gemini_key:
        raise ValueError("GEMINI_API_KEY nije podešen u Secrets.")

    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={gemini_key}"
    headers = {"Content-Type": "application/json"}
    
    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {"text": f"SISTEMSKO UPUTSTVO:\n{system_prompt}\n\nKORISNIČKO PITANJE:\n{user_query}"}
                ]
            }
        ],
        "generationConfig": {
            "temperature": 0.1,
            "maxOutputTokens": 1500
        }
    }
    
    res = requests.post(url, json=payload, headers=headers, timeout=30)
    res.raise_for_status()
    
    data = res.json()
    return data["candidates"][0]["content"]["parts"][0]["text"]


# --- GLAVNI RAG FLOW ---
def ask_birochat(user_query: str) -> Dict[str, Any]:
    internal_res = retrieve_internal_context(user_query)
    
    used_web_search = False
    web_sources = []

    # WEB PRETRAGA SE SE POKREĆE SAMO AKO NEMA ZAPOSLENIH I SCORE JE NIŽI OD 0.20
    has_matched_employees = len(internal_res["matched_employees"]) > 0
    
    if not has_matched_employees and (internal_res["max_score"] < 0.20 or not internal_res["sources"]):
        used_web_search = True
        web_res = search_web_tavily(user_query)
        web_sources = web_res["web_sources"]
        
        combined_context = (
            "⚠️ Napomena: Podaci nisu pronađeni u internoj bazi Biroa. Korišćena je pretraga sa interneta.\n\n"
            f"{web_res['web_context']}"
        )
    else:
        combined_context = f"INTERNI PODACI I PROFILI ZAPOSLENIH:\n{internal_res['context_str']}"

    # STROGO SISTEMSKO UPUTSTVO PROTIV GENERALIZACIJE
    system_prompt = (
        "Ti si BiroChat AI asistent za internek podatke Biroa i JP Srbijašume.\n\n"
        "VAŽNA PRAVILA ZA ODGOVARANJE:\n"
        "1. Ako korisnik pita ko obavlja neku funkciju (npr. 'Ko je zamenik direktora?', 'Ko je direktor?', 'Ko su blagajnici?'), "
        "IZRIČITO I ODMAH navedi tačna IMENA I PREZIMENA osoba navedenih u kontekstu!\n"
        "2. STROGO JE ZABRANJENO davati opšte definicije radnih mesta ili objašnjavati šta ta funkcija inače znači u teoriji (npr. 'Zamenik direktora je osoba koja pomaže...'). Korisnika zanimaju SAMO KONKRETNA LICA iz baze.\n"
        "3. Budi direktan, jasan i precizan. Ako ima više lica za istu funkciju (npr. dva zamenika direktora), navedi ih sve u listi.\n\n"
        f"PRILOŽENI KONTEKST IZ BAZE:\n{combined_context}"
    )

    used_provider = "Groq (LLaMA-3.3-70B)"
    
    try:
        answer = query_groq(system_prompt, user_query)
    except Exception as groq_err:
        print(f"⚠️ Groq greška ({groq_err}). Preusmeravam na Google Gemini Online...")
        try:
            used_provider = "Google Gemini 2.0 Flash (Online Free)"
            answer = query_gemini_online(system_prompt, user_query)
        except Exception as fallback_err:
            used_provider = "Greška"
            answer = f"❌ Greška pri komunikaciji sa AI modelima:\n- Groq: `{groq_err}`\n- Gemini: `{fallback_err}`"

    return {
        "answer": answer,
        "sources": internal_res["sources"],
        "web_sources": web_sources,
        "used_web": used_web_search,
        "provider": used_provider,
        "matched_employees": internal_res["matched_employees"]
    }


def query_biro_system(user_query: str) -> Tuple[str, List[Dict[str, Any]]]:
    res = ask_birochat(user_query)
    return res["answer"], res["matched_employees"]