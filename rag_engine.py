import os
import requests
from typing import Dict, Any, List
from qdrant_client import QdrantClient
from langchain_huggingface import HuggingFaceEmbeddings
from groq import Groq

# --- KONFIGURACIJA ---
QDRANT_URL = "https://09ffa5ef-2765-45c8-bfcf-29bc6bf90f08.eu-west-2-0.aws.cloud.qdrant.io"
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "TVOJ_QDRANT_API_KLJUC")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "TVOJ_GROQ_API_KLJUC")

OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "qwen2.5:7b"  # prilagodi lokalnom modelu koji imaš u Ollama-i

embed_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, check_compatibility=False)
groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None


def retrieve_context(query: str, top_k: int = 3) -> Dict[str, Any]:
    """Pretražuje Qdrant i priprema kontekst sa metapodacima o slikama i tabelama."""
    query_vector = embed_model.embed_query(query)
    
    # Ažurirana Qdrant query_points sintaksa
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
        doc_name = payload.get("document_name", "Neznan dokument")
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


def query_groq(system_prompt: str, user_query: str) -> str:
    """Poziva Groq API (Primary LLM)."""
    if not groq_client:
        raise ValueError("Groq API ključ nije podešen.")
        
    response = groq_client.chat.completions.create(
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
    """Poziva lokalni Ollama API ako Groq otkaže."""
    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query}
        ],
        "stream": False
    }
    res = requests.post(OLLAMA_URL, json=payload, timeout=60)
    res.raise_for_status()
    return res.json()["message"]["content"]


def ask_birochat(user_query: str) -> Dict[str, Any]:
    """Glavna RAG funkcija sa automatskim fallback-om."""
    retrieved = retrieve_context(user_query)
    
    system_prompt = (
        "Ti si BiroChat AI asistent. Odgovaraj precizno i profesionalno na srpskom jeziku.\n"
        "Koristi ISKLJUČIVO dole priloženi kontekst iz interne baze za formulisanje odgovora.\n"
        "Ako u kontekstu postoje informacije o prepoznatim zaposlenima ili priloženim tabelama, obavezno ih navedi.\n"
        "Ako kontekst ne sadrži odgovor, iskreno reci da nemaš te podatke u bazi.\n\n"
        f"KONTEKST IZ BAZE:\n{retrieved['full_context']}"
    )

    used_provider = "Groq (LLaMA-3.3-70B)"
    
    try:
        print("🚀 Šaljem upit na Groq API...")
        answer = query_groq(system_prompt, user_query)
    except Exception as e:
        print(f"⚠️ Groq greška ({e}). Preusmeravam na lokalni Ollama fallback...")
        try:
            used_provider = f"Lokalni Ollama ({OLLAMA_MODEL})"
            answer = query_ollama_fallback(system_prompt, user_query)
        except Exception as fallback_err:
            answer = f"❌ Greška pri generisanju odgovora: {fallback_err}"
            used_provider = "None"

    return {
        "answer": answer,
        "sources": retrieved["sources"],
        "provider": used_provider
    }


if __name__ == "__main__":
    test_q = "Pronađi sve analitičke kartice i proveri koji zaposleni se pominju u dokumentima."
    res = ask_birochat(test_q)
    
    print("\n" + "="*50)
    print(f"🤖 ODGOVOR ({res['provider']}):\n")
    print(res["answer"])
    print("\n📚 Izvori:", res["sources"])