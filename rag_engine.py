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


def is_person_query(query: str) -> bool:
    """Proverava da li korisnik postavlja pitanje u vezi sa zaposlenima ili rukovodstvom."""
    q = query.lower()
    person_triggers = [
        "ko je", "ko su", "direktor", "zamenik", "zamjenik", "blagajnik", 
        "geodeta", "zaposlen", "zaposleni", "radnik", "rukovodstvo", 
        "menadžment", "šef", "rukovodilac", "kadar", "slika", "fotografija", "profil"
    ]
    return any(trigger in q for trigger in person_triggers)


def filter_employees(query: str, all_employees: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Precizno filtrire zaposlene prema ulozi ili imenu iz upita."""
    q = query.lower()
    matched = {}

    is_asking_director = ("direktor" in q) and not ("zamenik" in q or "zamjenik" in q)
    is_asking_deputy = "zamenik" in q or "zamjenik" in q
    is_asking_cashier = "blagajnik" in q
    is_asking_surveyor = "geodeta" in q

    for emp in all_employees:
        name = emp.get("name", "")
        if not name:
            continue
        
        name_l = name.lower()
        role_l = emp.get("role", "").lower()
        is_dir = emp.get("is_director", False) or ("direktor" in role_l and "zamenik" not in role_l)
        is_dep = emp.get("is_deputy_director", False) or "zamenik" in role_l or "zamjenik" in role_l

        match = False
        if is_asking_director and is_dir and not is_dep:
            match = True
        elif is_asking_deputy and is_dep:
            match = True
        elif is_asking_cashier and "blagajnik" in role_l:
            match = True
        elif is_asking_surveyor and "geodeta" in role_l:
            match = True
        elif name_l in q or any(part in q for part in name_l.split() if len(part) > 3):
            match = True

        if match and name not in matched:
            matched[name] = emp

    return list(matched.values())


def retrieve_internal_context(query: str) -> Dict[str, Any]:
    embed_model = load_embedding_model()
    qdrant_client = get_qdrant_client()

    query_vector = embed_model.encode(query).tolist()
    is_person = is_person_query(query)

    formatted_context = []
    sources = []
    matched_employees = []

    # 1. AKO JE PITANJE O ZAPOSLENIMA - TRAŽE SE ISKLJUČIVO PROFILI
    if is_person:
        all_employees = []
        try:
            scroll_res, _ = qdrant_client.scroll(
                collection_name=COLLECTION_NAME,
                limit=100,
                with_payload=True,
                with_vectors=False
            )
            for point in scroll_res:
                p = point.payload or {}
                if p.get("doc_type") in ["employee", "employee_profile"]:
                    all_employees.append(p)
        except Exception as e:
            print(f"⚠️ Greška pri čitanju profila: {e}")

        matched_employees = filter_employees(query, all_employees)
        
        for emp in matched_employees:
            name = emp.get("name", "")
            role = emp.get("role", "")
            block = f"👤 ZAPOSLENI: {name}\n💼 FUNKCIJA: {role}"
            formatted_context.append(block)
            sources.append(f"Profil: {name}")

        return {
            "context_str": "\n---\n".join(formatted_context),
            "sources": list(set(sources)),
            "matched_employees": matched_employees,
            "has_data": len(matched_employees) > 0
        }

    # 2. AKO JE PITANJE O DOKUMENTIMA/OPŠTIM TEMAMA - ZAPOSLENI SE POTPUNO ISKLJUČUJU
    try:
        response = qdrant_client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            limit=10
        )
        results = response.points
    except Exception as e:
        print(f"⚠️ Qdrant greška: {e}")
        results = []

    SCORE_THRESHOLD = 0.38  # Prag ispod koga se dokument smatra nerelevantnim

    for point in results:
        payload = point.payload or {}
        doc_type = payload.get("doc_type", "document")

        # Preskoči zaposlene kada se traže opšti dokumenti
        if doc_type in ["employee", "employee_profile"]:
            continue

        if point.score >= SCORE_THRESHOLD:
            doc_name = payload.get("document_name", payload.get("name", "Dokument"))
            content = payload.get("markdown_content", payload.get("text", ""))[:2000]
            if content.strip():
                block = f"📄 DOKUMENT: {doc_name}\n📝 SADRŽAJ:\n{content}"
                formatted_context.append(block)
                sources.append(doc_name)

    return {
        "context_str": "\n---\n".join(formatted_context),
        "sources": list(set(sources)),
        "matched_employees": [],  # Uvek prazno za opšta pitanja
        "has_data": len(formatted_context) > 0
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
            web_blocks.append(f"🌐 WEB IZVOR: {result.get('title')}\n{result.get('content')}\n")
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
        max_tokens=1000
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
        "generationConfig": {"temperature": 0.1, "maxOutputTokens": 1000}
    }
    res = requests.post(url, json=payload, headers=headers, timeout=30)
    res.raise_for_status()
    return res.json()["candidates"][0]["content"]["parts"][0]["text"]


def ask_birochat(user_query: str) -> Dict[str, Any]:
    internal_res = retrieve_internal_context(user_query)

    used_web_search = False
    web_sources = []

    if not internal_res["has_data"]:
        system_prompt = (
            "Ti si BiroChat AI asistent za interne podatke Biroa.\n"
            "U internoj bazi podataka TRENUTNO NE POSTOJE traženi podaci ili dokumenti za ovo pitanje.\n"
            "Odgovori korisniku kratko i direktno da traženi podatak (npr. član ugovora, dijagram, dokument) "
            "nije pronađen u bazi Biroa. NIKADA nemoj izmišljati podatke ili navoditi imena zaposlenih!"
        )
    else:
        combined_context = internal_res["context_str"]
        system_prompt = (
            "Ti si BiroChat AI asistent za interne podatke Biroa.\n"
            "PRAVILA:\n"
            "1. Odgovaraj isključivo na osnovu priloženih podataka iz baze.\n"
            "2. Budi kratak, precizan i direktan.\n"
            "3. Ako su priloženi profili zaposlenih, navedi samo njihovo ime, prezimenu i funkciju.\n"
            "4. ZABRANJENO JE generisati opise fizičkog izgleda slika ili nabrajati ljude koji nisu traženi.\n\n"
            f"PODACI IZ BAZE:\n{combined_context}"
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