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

    def _generate_search_queries(self, user_question: str) -> list:
        """ Generiše više ciljanih upita za sigurniji obuhvat iz Qdrant-a """
        q_norm = normalize_text(user_question)
        queries = [user_question]

        # Ako se traže štampači/oprema, forsira povlačenje svih brand-ova i plotera
        if any(w in q_norm for w in ["stampac", "stampaci", "printer", "oprema", "ploter"]):
            queries.append("HP Designjet Canon TX Kyocera ploter štampač oprema toneri")

        # Ako se traži konkretan član (npr. član 14)
        article_match = re.search(r'(?:clan|član)\s*(\d+)', q_norm)
        if article_match:
            art_num = article_match.group(1)
            queries.append(f"Kolektivni ugovor Član {art_num}. clan {art_num}")
            queries.append(f"član {art_num} clan {art_num}")

        return queries

    def query(self, user_question: str) -> dict:
        context_texts = []
        sources = []
        
        q_norm = normalize_text(user_question)
        is_internal_query = any(w in q_norm for w in INTERNAL_WORDS)
        article_match = re.search(r'(?:clan|član)\s*(\d+)', q_norm)
        target_article = article_match.group(1) if article_match else None

        # 1. PRETRAGA LOKALNE DOKUMENTACIJE (QDRANT) - Multi-Query Engine
        try:
            search_queries = self._generate_search_queries(user_question)
            all_retrieved_docs = []
            seen_contents = set()

            for q_str in search_queries:
                docs = self.vector_store.similarity_search(q_str, k=12)
                for doc in docs:
                    content_head = doc.page_content.strip()[:100]
                    if content_head not in seen_contents:
                        seen_contents.add(content_head)
                        all_retrieved_docs.append(doc)

            extracted_chunks = []
            for doc in all_retrieved_docs:
                text_content = doc.page_content.strip()
                source_file = doc.metadata.get("source", "Lokalni dokument")
                extracted_chunks.append({
                    "text": text_content,
                    "source": source_file
                })

            # HIBRIDNO RE-RANGIRANJE ZA ČLANOVE UGOVORA
            if target_article:
                def score_chunk(chunk):
                    c_norm = normalize_text(chunk["text"])
                    if f"clan {target_article}" in c_norm or f"clan{target_article}" in c_norm:
                        return 100
                    return 0
                extracted_chunks.sort(key=score_chunk, reverse=True)

            # Sastavljanje finalnog konteksta za LLM
            for item in extracted_chunks[:10]:
                text_content = item["text"]
                if len(text_content) > 1500:
                    text_content = text_content[:1500] + "..."
                    
                if item["source"] not in sources:
                    sources.append(item["source"])
                
                block = f"Izvor: {item['source']}\n{text_content}"
                context_texts.append(block)

        except Exception as e:
            pass 
            
        # 2. PRETRAGA WEBA (TAVILY API) - Za opšta pitanja
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
            return {"answer": "Podatak nije dostupan u bazi dokumentacije.", "source_documents": []}
            
        context = "\n\n---\n\n".join(context_texts)
        if len(context) > 7000:
            context = context[:7000] + "\n...[Kontekst skraćen]"
        
        system_prompt = (
            "Ti si BiroChat, korporativni asistent za pretragu dokumentacije Biroa za planiranje i projektovanje u šumarstvu.\n"
            "Odgovori precizno koristeći ISKLJUČIVO navedeni kontekst.\n\n"
            "VAŽNA PRAVILA ZA STRUKTURU ODGOVORA:\n"
            "1. KADA JE PITANJE 'Koji štampači se koriste u Birou?' ili opšte o štampačima/opremi:\n"
            "   - Izvuci i navedi SAMO čiste nazive modela štampača i plotera (npr. HP Designjet 800PS, Canon TX-3000, Kyocera FS-9530dn, Kyocera M3655idn, Kyocera P2040dn).\n"
            "   - NIKADA nemoj navoditi šifre tonera (poput TK-710, HP C4844A) niti količine komada kada korisnik pita koji se štampači koriste.\n"
            "2. KADA SE TRAŽI KONKRETAN ČLAN (npr. Član 14 Kolektivnog ugovora):\n"
            "   - Pronađi taj član u kontekstu i navedi njegov pun tekst ili detaljno sumiraj sve njegove odredbe.\n"
            "3. Ako podatak zaista ne postoji u kontekstu, odgovori sa 'Podatak nije dostupan.'"
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