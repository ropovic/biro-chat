import os
import re
import requests
from typing import Dict, Any, List, Tuple
import streamlit as st
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from sentence_transformers import SentenceTransformer
from groq import Groq


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
GEMINI_API_KEY = get_config("GEMINI_API_KEY", "")
COLLECTION_NAME = get_config("COLLECTION_NAME", "Baza_biro")


@st.cache_resource(show_spinner=False)
def load_embedding_model():
    # Keshira model u memoriji da ne bi punio RAM pri svakom upitu
    return SentenceTransformer("BAAI/bge-m3")


@st.cache_resource(show_spinner=False)
def get_qdrant_client():
    return QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, check_compatibility=False)


def get_groq_client() -> Groq | None:
    key = get_config("GROQ_API_KEY", "")
    return Groq(api_key=key) if key else None


def is_person_query(query: str) -> bool:
    q = query.lower()
    person_triggers = [
        "ko je", "ko su", "direktor", "zamenik", "zamjenik", "blagajnik", 
        "geodeta", "zaposlen", "zaposleni", "radnik", "rukovodstvo", 
        "menadžment", "šef", "rukovodilac", "kadar", "slika", "fotografija", "profil"
    ]
    return any(trigger in q for trigger in person_triggers)


def retrieve_internal_context(query: str) -> Dict[str, Any]:
    embed_model = load_embedding_model()
    qdrant_client = get_qdrant_client()

    query_vector = embed_model.encode(query).tolist()
    is_person = is_person_query(query)

    formatted_context = []
    sources = []
    matched_employees = []
    seen_texts = set()

    # 1. AKO SE TRAŽE ZAPOSLENI (KORISTI SE EKSPLICITNI FILTER)
    if is_person:
        try:
            scroll_res, _ = qdrant_client.scroll(
                collection_name=COLLECTION_NAME,
                scroll_filter=Filter(
                    should=[
                        FieldCondition(key="doc_type", match=MatchValue(value="employee")),
                        FieldCondition(key="doc_type", match=MatchValue(value="employee_profile"))
                    ]
                ),
                limit=100,
                with_payload=True,
                with_vectors=False
            )
            
            q_low = query.lower()
            for point in scroll_res:
                p = point.payload or {}
                name = p.get("name", "")
                role = p.get("role", "")
                
                # Provera uklapanja uloge/imena
                is_dir = "direktor" in role.lower() and "zamenik" not in role.lower()
                is_dep = "zamenik" in role.lower() or "zamjenik" in role.lower()

                match = False
                if "direktor" in q_low and not ("zamenik" in q_low or "zamjenik" in q_low) and is_dir:
                    match = True
                elif ("zamenik" in q_low or "zamjenik" in q_low) and is_dep:
                    match = True
                elif any(part.lower() in q_low for part in name.split() if len(part) > 2):
                    match = True
                elif "zaposlen" in q_low or "rukovodstvo" in q_low:
                    match = True

                if match:
                    matched_employees.append(p)
                    block = f"👤 ZAPOSLENI: {name}\n💼 FUNKCIJA: {role}"
                    if block not in seen_texts:
                        formatted_context.append(block)
                        seen_texts.add(block)
                        sources.append(f"Profil: {name}")

        except Exception as e:
            print(f"⚠️ Qdrant scroll greška: {e}")

        return {
            "context_str": "\n---\n".join(formatted_context),
            "sources": list(set(sources)),
            "matched_employees": matched_employees,
            "has_data": len(matched_employees) > 0
        }

    # 2. PRETRAGA DOKUMENATA (POGŠ, ŠTAMPAČI, UGOVORI)
    try:
        response = qdrant_client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            limit=15
        )
        results = response.points
    except Exception as e:
        print(f"⚠️ Qdrant query greška: {e}")
        results = []

    # Dinamički prag relevantnosti
    q_low = query.lower()
    is_specific = any(w in q_low for w in ["pogš", "osnov", "štampač", "toner", "kolektivni", "ugovor", "član"])
    SCORE_THRESHOLD = 0.15 if is_specific else 0.22

    for point in results:
        payload = point.payload or {}
        doc_type = payload.get("doc_type", "document")

        if doc_type in ["employee", "employee_profile"]:
            continue

        if point.score >= SCORE_THRESHOLD:
            doc_name = payload.get("document_name", payload.get("name", payload.get("source", "Dokument")))
            content = payload.get("markdown_content", payload.get("text", ""))[:2000]

            if content.strip() and content not in seen_texts:
                block = f"📄 DOKUMENT: {doc_name}\n📝 SADRŽAJ:\n{content}"
                formatted_context.append(block)
                seen_texts.add(content)
                sources.append(doc_name)

    return {
        "context_str": "\n---\n".join(formatted_context),
        "sources": list(set(sources)),
        "matched_employees": [],
        "has_data": len(formatted_context) > 0
    }


def query_groq(system_prompt: str, user_query: str) -> str:
    client = get_groq_client()
    if not client:
        raise ValueError("GROQ_API_KEY nije podešen.")

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query}
        ],
        temperature=0.0,
        max_tokens=800,
        timeout=10.0  # Ograničeno čakanje na 10 sekundi
    )
    return response.choices[0].message.content


def query_gemini_online(system_prompt: str, user_query: str) -> str:
    gemini_key = get_config("GEMINI_API_KEY", "")
    if not gemini_key:
        raise ValueError("GEMINI_API_KEY nije podešen.")

    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={gemini_key}"
    headers = {"Content-Type": "application/json"}
    payload = {
        "contents": [{"role": "user", "parts": [{"text": f"{system_prompt}\n\nKORISNIK: {user_query}"}]}],
        "generationConfig": {"temperature": 0.0, "maxOutputTokens": 800}
    }
    res = requests.post(url, json=payload, headers=headers, timeout=10)
    res.raise_for_status()
    return res.json()["candidates"][0]["content"]["parts"][0]["text"]


def ask_birochat(user_query: str) -> Dict[str, Any]:
    internal_res = retrieve_internal_context(user_query)

    if not internal_res["has_data"]:
        system_prompt = (
            "Ti si BiroChat AI asistent.\n"
            "U bazi podataka TRENUTNO NE POSTOJE traženi dokumenti ili podaci za ovo pitanje.\n"
            "Odgovori kratko da podatak nije pronađen u bazi."
        )
    else:
        combined_context = internal_res["context_str"]
        system_prompt = (
            "Ti si BiroChat AI asistent za interne podatke Biroa.\n"
            "Odgovaraj kratko i precizno isključivo na osnovu teksta ispod.\n\n"
            f"PODACI IZ BAZE:\n{combined_context}"
        )

    used_provider = "Groq"
    try:
        answer = query_groq(system_prompt, user_query)
    except Exception as e:
        print(f"Groq timeout/err: {e}, prelazak na Gemini...")
        try:
            used_provider = "Gemini"
            answer = query_gemini_online(system_prompt, user_query)
        except Exception as e2:
            answer = f"Greška pri dobijanju odgovora: {e2}"

    return {
        "answer": answer,
        "sources": internal_res["sources"],
        "provider": used_provider,
        "matched_employees": internal_res["matched_employees"]
    }