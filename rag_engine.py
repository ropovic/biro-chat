import os
import requests
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from groq import Groq

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
        self.tavily_api_key = os.getenv("TAVILY_API_KEY") # Obavezno setovati ovaj ključ u .env

    def query(self, user_question: str) -> dict:
        context_texts = []
        sources = []
        
        # 1. PRETRAGA LOKALNE DOKUMENTACIJE (QDRANT)
        try:
            docs = self.vector_store.similarity_search(user_question, k=3)
            for doc in docs:
                text_content = doc.page_content.strip()
                if len(text_content) > 800:
                    text_content = text_content[:800] + "..."
                    
                source_file = doc.metadata.get("source", "Lokalni dokument")
                if source_file not in sources:
                    sources.append(source_file)
                
                block = f"Izvor: {source_file}\n{text_content}"
                context_texts.append(block)
        except Exception as e:
            pass # Nastavlja dalje ukoliko je Qdrant baza trenutno nedostupna
            
        # 2. PRETRAGA WEBA (TAVILY API)
        if self.tavily_api_key:
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
            return {"answer": "Nažalost, nisam pronašao relevantne informacije ni u bazi ni na webu.", "source_documents": []}
            
        context = "\n\n---\n\n".join(context_texts)
        
        # Limit tokena za Groq
        if len(context) > 3500:
            context = context[:3500] + "\n...[Kontekst skraćen]"
        
        system_prompt = (
            "Ti si BiroChat, korporativni asistent. "
            "Odgovori kratko i precizno na pitanje koristeći ISKLJUČIVO navedeni kontekst (koji uključuje lokalne dokumente i/ili web pretragu). "
            "Ako u kontekstu nema traženog podatka, napiši da podatak nije dostupan."
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
                max_tokens=400
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