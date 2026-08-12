from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from groq import Groq
import config

qdrant = QdrantClient(url=config.QDRANT_URL, api_key=config.QDRANT_API_KEY)
embedder = SentenceTransformer("all-MiniLM-L6-v2")
groq_client = Groq(api_key=config.GROQ_API_KEY)

def query_biro_system(user_query: str):
    # 1. Embedding korisničkog pitanja
    query_vector = embedder.encode(user_query).tolist()
    
    # 2. Vektorska pretraga u Qdrant-u
    search_results = qdrant.search(
        collection_name=config.COLLECTION_NAME,
        query_vector=query_vector,
        limit=5
    )
    
    context_blocks = []
    matched_employees = []
    
    for res in search_results:
        payload = res.payload
        context_blocks.append(payload.get("text", ""))
        
        # Ako je rezulat profil zaposlenog, sačuvaj metapodatke za UI
        if payload.get("doc_type") == "employee_profile":
            matched_employees.append(payload)

    context_str = "\n---\n".join(context_blocks)
    
    # 3. Formiranje prompta za Groq (LLaMA-3.3-70B)
    system_prompt = (
        "Ti si službeni AI asistent Biroa. Odgovaraj tačno i precizno na srpskom jeziku "
        "na osnovu priloženih internih dokumenata i profila zaposlenih.\n"
        "Ako su u kontekstu navedeni podaci o direktoru i zaposlenima, jasno ih navedi."
    )
    
    user_prompt = f"Kontekst iz baze:\n{context_str}\n\nPitanje korisnika: {user_query}"
    
    # 4. Poziv Groq API-ja
    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.2
    )
    
    llm_answer = response.choices[0].message.content
    return llm_answer, matched_employees