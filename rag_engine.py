import os
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from groq import Groq

class RAGEngine:
    def __init__(self):
        # Parametri za Qdrant Cloud
        self.qdrant_url = "https://09ffa5ef-2765-45c8-bfcf-29bc6bf90f08.eu-west-2-0.aws.cloud.qdrant.io"
        self.qdrant_api_key = os.getenv("QDRANT_API_KEY", "VAŠ_API_KLJUČ_OVDE")
        self.collection_name = "Baza_biro"
        
        # Sentence Transformer model za embedding
        self.encoder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        
        # Qdrant klijent
        self.qdrant_client = QdrantClient(
            url=self.qdrant_url,
            api_key=self.qdrant_api_key,
            check_compatibility=False
        )
        
        # Groq klijent
        self.groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.llm_model = "llama3-8b-8192"

    def query(self, user_question: str) -> dict:
        """Pretražuje Bazu_biro i vraća rečnik sa 'answer' i 'source_documents'."""
        
        # 1. Generisanje vektora
        query_vector = self.encoder.encode(user_question).tolist()
        
        try:
            # 2. Pretraga u Qdrant Cloud bazi
            search_results = self.qdrant_client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=5
            )
        except Exception as e:
            return {"answer": f"⚠️ Greška prilikom komunikacije sa Qdrant bazom: {e}", "source_documents": []}
            
        if not search_results:
            return {"answer": "Nažalost, nisam pronašao relevantne informacije u bazi podataka.", "source_documents": []}
            
        # 3. Ekstrakcija teksta iz Langchain payloada ('page_content')
        context_texts = []
        sources = []
        for hit in search_results:
            payload = hit.payload or {}
            text_content = payload.get("page_content") or payload.get("text")
            
            if text_content:
                metadata = payload.get("metadata", {})
                source_file = metadata.get("source", "Dokument")
                sources.append(source_file)
                
                block = f"📄 Izvor: {source_file}\nSadržaj:\n{text_content}"
                context_texts.append(block)
                
        if not context_texts:
            return {"answer": "Pronađeni su rezultati u bazi, ali tekstualni sadržaj nije mogao biti pročitan.", "source_documents": []}
            
        context = "\n---\n".join(context_texts)
        
        # 4. Sistemski prompt
        system_prompt = (
            "Ti si BiroChat, korporativni asistent za pretragu dokumentacije. "
            "Odgovori precizno na korisničko pitanje koristeći ISKLJUČIVO priloženi kontekst iz baze. "
            "Ako podatak ne postoji u kontekstu, jasno reci da ga nema. Nemoj izmišljati."
        )
        
        user_prompt = f"Kontekst iz baze:\n{context}\n\nPitanje: {user_question}"
        
        # 5. Poziv Groq API-ja
        try:
            response = self.groq_client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                model=self.llm_model,
                temperature=0.1,
                max_tokens=1024
            )
            answer_text = response.choices[0].message.content
            
            # Vraćamo rečnik koji odgovara očekivanju u app.py
            return {
                "answer": answer_text,
                "source_documents": context_texts,
                "sources": sources
            }
        except Exception as e:
            return {"answer": f"⚠️ Greška prilikom generisanja odgovora od strane LLM-a: {e}", "source_documents": []}

# Globalna instanca klase
_engine_instance = RAGEngine()

# Funkcija koju app.py očekuje i uvozi
def ask_birochat(question: str) -> dict:
    """Omotač koji vraća dict sa odgovorom."""
    return _engine_instance.query(question)