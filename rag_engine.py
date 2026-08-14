import os
import re
import requests
import unicodedata
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from groq import Groq

def normalize_text(text: str) -> str:
    if not text: return ""
    text = text.lower().replace("đ", "dj")
    nfkd = unicodedata.normalize('NFKD', text)
    return "".join([c for c in nfkd if not unicodedata.combining(c)])

INTERNAL_WORDS = [
    "biro", "ugovor", "kolektivni", "stampac", "toner", "zaposlen", 
    "oprema", "clan", "pravilnik", "srbijasum", "direktor", "odmor", 
    "radni", "katalog", "ploter", "hp", "kyocera", "canon"
]

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

    def _expand_query(self, query: str) -> str:
        """ Proširuje sinonime da vektorska baza lakše pronađe sve štampače/plotere """
        q_norm = normalize_text(query)
        expanded = query
        if any(w in q_norm for w in ["stampac", "stampaci", "printer", "oprema"]):
            expanded += " ploter hp800 hp designjet kyocera canon mf uređaj"
        return expanded

    def query(self, user_question: str) -> dict:
        context_texts = []
        sources = []
        
        q_norm = normalize_text(user_question)
        is_internal_query = any(w in q_norm for w in INTERNAL_WORDS)
        
        # Detekcija specifičnog člana (npr. "clan 14" ili "član 14")
        article_match = re.search(r'(?:clan|član)\s*(\d+)', q_norm)
        target_article = article_match.group(1) if article_match else None

        # 1. PRETRAGA LOKALNE DOKUMENTACIJE (QDRANT) - Povećano k=15
        try:
            search_query = self._expand_query(user_question)
            docs = self.vector_store.similarity_search(search_query, k=15)
            
            extracted_chunks = []
            for doc in docs:
                text_content = doc.page_content.strip()
                source_file = doc.metadata.get("source", "Lokalni dokument")
                extracted_chunks.append({
                    "text": text_content,
                    "source": source_file
                })

            # HIBRIDNO RE-RANGIRANJE ZA ČLANOVE: Ako se traži Član X, poguraj ga na vrh!
            if target_article:
                def score_chunk(chunk):
                    c_norm = normalize_text(chunk["text"])
                    if f"clan {target_article}" in c_norm or f"clan{target_article}" in c_norm:
                        return 100
                    return 0
                extracted_chunks.sort(key=score_chunk, reverse=True)

            # Sastavljanje konteksta
            for item in extracted_chunks[:8]:  # Uzimamo najboljih 8 nakon re-rangiranja
                text_content = item["text"]
                if len(text_content) > 1500:
                    text_content = text_content[:1500] + "..."
                    
                if item["source"] not in sources:
                    sources.append(item["source"])
                
                block = f"Izvor: {item['source']}\n{text_content}"
                context_texts.append(block)

        except Exception as e:
            pass 
            
        # 2. PRETRAGA WEBA (TAVILY API) - Samo za opšta spoljna pitanja
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
            return {"answer": "Nažalost, podatak nije pronađen u dokumentima baze.", "source_documents": []}
            
        context = "\n\n---\n\n".join(context_texts)
        if len(context) > 6000:
            context = context[:6000] + "\n...[Kontekst skraćen]"
        
        system_prompt = (
            "Ti si BiroChat, korporativni asistent za pretragu dokumentacije. "
            "Odgovori precizno koristeći ISKLJUČIVO navedeni kontekst. "
            "Uvek navedi SVE štampače, plotere i opremu koji se pominju u kontekstu (uključujući HP, Kyocera, Canon). "
            "Ako se traži konkretan član ugovora, prepiši ili detaljno sumiraj taj član iz konteksta. "
            "Ako u kontekstu nema traženih podataka, jasno navedi da podatak nije dostupan."
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
                max_tokens=800
            )
            answer_text = response.choices[0].message.content
            
            return {
                "answer": answer_text,
                "source_documents": context_texts,
                "sources": sources
            }
        except Exception as e:
            return {"answer": f"⚠️ Greška prilikom generisanja odgovora: {e}", "source_documents": []}

_engine_instance = RAGEngine()

def ask_birochat(question: str) -> dict:
    return _engine_instance.query(question)