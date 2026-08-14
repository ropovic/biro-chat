import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from groq import Groq

class RAGEngine:
    def __init__(self):
        # Parametri za Qdrant Cloud
        self.qdrant_url = "https://09ffa5ef-2765-45c8-bfcf-29bc6bf90f08.eu-west-2-0.aws.cloud.qdrant.io"
        self.qdrant_api_key = os.getenv("QDRANT_API_KEY", "VAŠ_API_KLJUČ_OVDE") # Obavezno unesite ključ
        self.collection_name = "Baza_biro"
        
        # 1. Inicijalizacija istog embedding modela korišćenog pri indeksiranju
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}, 
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # 2. Povezivanje na Qdrant Cloud
        self.client = QdrantClient(
            url=self.qdrant_url,
            api_key=self.qdrant_api_key,
            check_compatibility=False
        )
        
        # 3. Langchain omotač za pretragu (eliminiše 'search' attribute error)
        self.vector_store = QdrantVectorStore(
            client=self.client,
            collection_name=self.collection_name,
            embedding=self.embeddings
        )
        
        # 4. Groq klijent
        self.groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.llm_model = "llama-3.1-8b-instant"

    def query(self, user_question: str) -> dict:
        """Pretražuje Bazu_biro preko Langchain-a i vraća rečnik."""
        
        try:
            # Pretraga preko Langchain-a (metoda similarity_search)
            docs = self.vector_store.similarity_search(user_question, k=5)
        except Exception as e:
            return {"answer": f"⚠️ Greška prilikom pretrage: {e}", "source_documents": []}
            
        if not docs:
            return {"answer": "Nažalost, nisam pronašao relevantne informacije u bazi podataka.", "source_documents": []}
            
        context_texts = []
        sources = []
        
        # Langchain automatski kreira objekte sa .page_content i .metadata
        for doc in docs:
            text_content = doc.page_content
            source_file = doc.metadata.get("source", "Nepoznat dokument")
            
            # Filtriranje kako ne bismo dodavali iste izvore više puta u listu
            if source_file not in sources:
                sources.append(source_file)
            
            block = f"📄 Izvor: {source_file}\nSadržaj:\n{text_content}"
            context_texts.append(block)
            
        context = "\n---\n".join(context_texts)
        
        # Sistemski prompt
        system_prompt = (
            "Ti si BiroChat, korporativni asistent za pretragu dokumentacije. "
            "Odgovori precizno na korisničko pitanje koristeći ISKLJUČIVO priloženi kontekst iz baze. "
            "Ako podatak ne postoji u kontekstu, jasno reci da ga nema. Nemoj izmišljati."
        )
        
        user_prompt = f"Kontekst iz baze:\n{context}\n\nPitanje: {user_question}"
        
        # Poziv Groq API-ja
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
            
            return {
                "answer": answer_text,
                "source_documents": context_texts,
                "sources": sources
            }
        except Exception as e:
            return {"answer": f"⚠️ Greška prilikom generisanja odgovora od strane LLM-a: {e}", "source_documents": []}

# Globalna instanca klase
_engine_instance = RAGEngine()

# Funkcija koju app.py očekuje
def ask_birochat(question: str) -> dict:
    return _engine_instance.query(question)