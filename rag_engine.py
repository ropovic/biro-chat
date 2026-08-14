import os
import requests
import unicodedata
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from groq import Groq

def normalize_text(text: str) -> str:
    text = text.lower().replace("đ", "dj")
    nfkd = unicodedata.normalize('NFKD', text)
    return "".join([c for c in nfkd if not unicodedata.combining(c)])

# Ključne reči koje označavaju da se traži lokalni dokument
INTERNAL_WORDS = ["biro", "ugovor", "kolektivni", "stampac", "toner", "zaposlen", "oprema", "clan", "pravilnik", "srbijasum", "direktor", "odmor", "radni", "katalog"]

class RAGEngine:
    def __init__(self):
        self.qdrant_url = "https://09ffa5ef-2765-45c8-bfcf-29bc6bf90f08.eu-west-2-0.aws.cloud.qdrant.io"
        self.qdrant_api_key = os.getenv("QDRANT_API_KEY", "TVOJ_QDRANT_API_KLJUČ")
        self.collection_name = "Baza_biro"
        
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            model_kwargs={'device': 'cpu'}, 
            encode_kwargs={'normalize_embeddings': True}
        )
        
        self.client = QdrantClient(
            url=self.qdrant_url,
            api_key=self.qdrant_api_key,
            check_compatibility=False
        )
        
        self.vector_store = QdrantVectorStore(
            client=self.client,
            collection_name=self.collection_name,
            embedding=self.embeddings
        )
        
        self.groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.llm_model = "llama-3.1-8b-instant"
        self.tavily_api_key = os.getenv("TAVILY_API_KEY")

    def query(self, user_question: str) -> dict:
        context_texts = []
        sources = []
        
        # Provera da li je upit isključivo interni
        q_norm = normalize_text(user_question)
        is_internal_query = any(w in q_norm for w in INTERNAL_WORDS)
        
        # 1. PRETRAGA LOKALNE DOKUMENTACIJE (QDRANT) - Povećano k=8
        try:
            docs = self.vector_store.similarity_search(user_question, k=8)
            for doc in docs:
                text_content = doc.page_content.strip()
                # Skraćujemo isečke na optimalnih 600 karaktera da stane HP800 i sve ostalo
                if len(text_content) > 600:
                    text_content = text_content[:600] + "..."
                    
                source_file = doc.metadata.get("source", "Lokalni dokument")
                if source_file not in sources:
                    sources.append(source_file)
                
                block = f"Izvor: {source_file}\n{text_content}"
                context_texts.append(block)
        except Exception as e:
            pass 
            
        # 2. PRETRAGA WEBA (TAVILY API) - Samo ako upit nije interni
        if self.tavily_api_key and not is_internal_query:
            try:
                tavily_resp = requests.post(
                    "https://api.tavily.com/search",
                    json={
                        "api_key": self.tavily_api_key, 
                        "query": user_question, 
                        "search_depth": "basic", 
                        "include_answer": True, 
                        "max_results": 2
                    },
                    timeout=5
                )
                if tavily_resp.status_code == 200:
                    data = tavily_resp.json()
                    tavily_answer = data.get("answer", "")
                    
                    if not tavily_answer:
                        snippets = [res["content"] for res in data.get("results", [])]
                        tavily_answer = " ".join(snippets)
                    
                    if tavily_answer:
                        context_texts.append(f"Izvor: Web Pretraga (Tavily)\n{tavily_answer}")
                        if "Web Pretraga (Tavily)" not in sources:
                            sources.append("Web Pretraga (Tavily)")
            except Exception as e:
                pass

        if not context_texts:
            return {"answer": "Nažalost, nisam pronašao relevantne informacije u dostupnim izvorima.", "source_documents": []}
            
        context = "\n\n---\n\n".join(context_texts)
        
        # Ograničenje za stabilnost tokena (4500 karaktera)
        if len(context) > 4500:
            context = context[:4500] + "\n...[Kontekst skraćen]"
        
        system_prompt = (
            "Ti si BiroChat, korporativni asistent za pretragu dokumentacije. "
            "Odgovori precizno koristeći ISKLJUČIVO navedeni kontekst. "
            "Ako u kontekstu nema odgovora, reci da podatak nije dostupan. "
            "Nikada ne izmišljaj članove zakona, pravilnika ili liste opreme ukoliko ih nema u kontekstu."
        )
        
        user_prompt = f"Kontekst:\n{context}\n\nPitanje: {user_question}"
        
        try:
            response = self.groq_client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                model=self.llm_model,
                temperature=0.0,
                max_tokens=500
            )
            answer_text = response.choices[0].message.content
            
            return {
                "answer": answer_text,
                "source_documents": context_texts,
                "sources": sources
            }
        except Exception as e:
            return {"answer": f"⚠️ Greška prilikom generisanja odgovora od strane LLM-a: {e}", "source_documents": []}

_engine_instance = RAGEngine()

def ask_birochat(question: str) -> dict:
    return _engine_instance.query(question)