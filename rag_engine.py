import os
import requests
from typing import Dict, Any, List, Tuple
import streamlit as st
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from groq import Groq
from tavily import TavilyClient


def get_config(key: str, default: str = "") -> str:
    try:
        if key in st.secrets:
            return str(st.secrets[key])
    except Exception:
        pass
    return os.getenv(key, default)


QDRANT_URL = get_config("QDRANT_URL", "https://09ffa5ef-2765-45c8-bfcf-29bc6bf90f08.eu-west-2-0.aws.cloud.qdrant.io")
QDRANT_API_KEY = get_config("QDRANT_API_KEY", "")
GROQ_API_KEY = get_config("GROQ_API_KEY", "")
TAVILY_API_KEY = get_config("TAVILY_API_KEY", "")
GEMINI_API_KEY = get_config("GEMINI_API_KEY", "")
COLLECTION_NAME = get_config("COLLECTION_NAME", "biro_documents")


@st.cache_resource(show_spinner=False)
def load_embedding_model():
    return SentenceTransformer("BAAI/bge-m3")


@st.cache_resource(show_spinner=False)
def get_qdrant_client():
    return QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, check_compatibility=False)


def get_groq_client() -> Groq | None:
    key = get_config("GROQ_API_KEY", "")
    return Groq(api_key=key) if key else None


def retrieve_internal_context(query: str, top_k: int = 8) -> Dict[str, Any]:
    embed_model = load_embedding_model()
    qdrant_client = get_qdrant_client()

    query_vector = embed_model.encode(query).tolist()
    
    # 1. Vektorska pretraga
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

    # 2. Direktna provera kartica zaposlenih (garantuje pronalazak profila)
    employee_profiles = []
    try:
        scroll_res, _ = qdrant_client.scroll(
            collection_name=COLLECTION_NAME,
            limit=100,
            with_payload=True,
            with_vectors=False
        )
        for point in scroll_res:
            p = point.payload or {}
            if p.get("doc_type") in ["employee", "employee_profile", "logo"]:
                employee_profiles.append(p)
    except Exception as e:
        print(f"⚠️ Greška pri preuzimanju profila: {e}")

    formatted_context = []
    sources = []
    matched_employees = []
    max_score = 0.0

    # Prvo obradi rezultate vektorske pretrage
    for res in results:
        if res.score > max_score:
            max_score = res.score

        payload = res.payload or {}
        doc_type = payload.get("doc_type", "document")
        
        if doc_type in ["employee", "employee_profile", "logo"]:
            if payload not in matched_employees:
                matched_employees.append(payload)
            name = payload.get("name", "Nepoznato")
            role = payload.get("role", "Zaposleni")
            desc = payload.get("description", "")
            
            block = f"👤 PROFIL ZAPOSLENOG: {name}\n💼 FUNKCIJA: {role}\n📝 OPIS: {desc}\n"
            formatted_context.append(block)
            sources.append(f"Profil: {name}")
            continue

        doc_name = payload.get("document_name", payload.get("name", "Dokument"))
        content = payload.get("markdown_content", payload.get("text", ""))[:2000]
        block = f"📄 DOKUMENT: {doc_name}\n📝 SADRŽAJ:\n{content}\n"
        formatted_context.append(block)
        sources.append(doc_name)

    # Ako se upit odnosi na funkcije/uloge, a vektorska pretraga nije izvučela odgovarajući profil
    query_lower = query.lower()
    for emp in employee_profiles:
        role_lower = emp.get("role", "").lower()
        desc_lower = emp.get("description", "").lower()
        name_lower = emp.get("name", "").lower()

        is_match = False
        if "direktor" in query_lower and ("direktor" in role_lower or emp.get("is_director")):
            if "zamenik" not in query_lower and "zamjenik" not in query_lower:
                if not emp.get("is_deputy_director") and "zamenik" not in role_lower:
                    is_match = True
            else:
                if emp.get("is_deputy_director") or "zamenik" in role_lower:
                    is_match = True
        elif "zamenik" in query_lower and ("zamenik" in role_lower or emp.get("is_deputy_director")):
            is_match = True
        elif "blagajnik" in query_lower and "blagajnik" in role_lower:
            is_match = True
        elif "geodeta" in query_lower and "geodeta" in role_lower:
            is_match = True
        elif name_lower and name_lower in query_lower:
            is_match = True

        if is_match and emp not in matched_employees:
            matched_employees.append(emp)
            block = f"👤 PROFIL ZAPOSLENOG: {emp.get('name')}\n💼 FUNKCIJA: {emp.get('role')}\n📝 OPIS: {emp.get('description')}\n"
            formatted_context.append(block)
            sources.append(f"Profil: {emp.get('name')}")

    return {
        "context_str": "\n" + "="*20 + "\n" + "\n=".join(formatted_context),
        "sources": list(set(sources)),
        "max_score": max_score,
        "matched_employees": matched_employees
    }


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
            web_blocks.append(f"🌐 WEB: {result.get('title')}\n{result.get('content')}\n")
            web_sources.append(result.get("url"))
        return {"web_context": "\n".join(web_blocks), "web_sources": web_sources}
    except Exception as e:
        print(f"⚠️ Tavily greška: {e}")
        return {"web_context": "", "web_sources": []}


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
        "contents": [{"role": "user", "parts": [{"text": f"{system_prompt}\n\nKORISNIK: {user_query}"}]}],
        "generationConfig": {"temperature": 0.1, "maxOutputTokens": 1500}
    }
    res = requests.post(url, json=payload, headers=headers, timeout=30)
    res.raise_for_status()
    return res.json()["candidates"][0]["content"]["parts"][0]["text"]


def ask_birochat(user_query: str) -> Dict[str, Any]:
    internal_res = retrieve_internal_context(user_query)
    
    has_matched_employees = len(internal_res["matched_employees"]) > 0
    used_web_search = False
    web_sources = []

    # Web pretraga se pokreće isključivo ako nema nijednog zaposlenog i skor je ispod 0.15
    if not has_matched_employees and (internal_res["max_score"] < 0.15 or not internal_res["sources"]):
        used_web_search = True
        web_res = search_web_tavily(user_query)
        web_sources = web_res["web_sources"]
        combined_context = f"WEB PODACI:\n{web_res['web_context']}"
    else:
        combined_context = f"INTERNI PODACI I PROFILI ZAPOSLENIH:\n{internal_res['context_str']}"

    system_prompt = (
        "Ti si BiroChat AI asistent za Biro i JP Srbijašume.\n"
        "PRAVILA:\n"
        "1. Odmah navedi tačno ime i prezime osobe ili osoba navedenih u kontekstu za traženu funkciju.\n"
        "2. STROGO JE ZABRANJENO pisati opšte definicije poslova ili rečničke opise radnih mesta.\n"
        "3. Odgovaraj kratko, tačno i direktno.\n\n"
        f"KONTEKST:\n{combined_context}"
    )

    used_provider = "Groq (LLaMA-3.3-70B)"
    try:
        answer = query_groq(system_prompt, user_query)
    except Exception as groq_err:
        print(f"Groq greška: {groq_err}. Prelazak na Gemini...")
        try:
            used_provider = "Google Gemini 2.0 Flash"
            answer = query_gemini_online(system_prompt, user_query)
        except Exception as fallback_err:
            used_provider = "Greška"
            answer = f"Greška AI modela: {groq_err} | {fallback_err}"

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