import os
import re
import requests
import unicodedata
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from groq import Groq

def cyrillic_to_latin(text: str) -> str:
    cyr_map = {
        'а':'a', 'б':'b', 'в':'v', 'г':'g', 'д':'d', 'ђ':'dj', 'е':'e', 'ж':'z',
        'з':'z', 'и':'i', 'ј':'j', 'к':'k', 'л':'l', 'љ':'lj', 'м':'m', 'н':'n',
        'њ':'nj', 'о':'o', 'п':'p', 'р':'r', 'с':'s', 'т':'t', 'ћ':'c', 'у':'u',
        'ф':'f', 'х':'h', 'ц':'c', 'ч':'c', 'џ':'dz', 'ш':'s',
        'А':'A', 'Б':'B', 'В':'V', 'Г':'G', 'Д':'D', 'Ђ':'Dj', 'Е':'E', 'Ж':'Z',
        'З':'Z', 'И':'I', 'Ј':'J', 'К':'K', 'Л':'L', 'Љ':'Lj', 'М':'M', 'Н':'N',
        'Њ':'Nj', 'О':'O', 'П':'P', 'Р':'R', 'С':'S', 'Т':'T', 'Ћ':'C', 'У':'U',
        'Ф':'F', 'Х':'H', 'Ц':'C', 'Ч':'C', 'Џ':'Dz', 'Ш':'S'
    }
    for cyr, lat in cyr_map.items():
        text = text.replace(cyr, lat)
    return text

def normalize_text(text: str) -> str:
    if not text: return ""
    text = cyrillic_to_latin(text).lower().replace("đ", "dj")
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
        q_norm = normalize_text(user_question)
        queries = [user_question]

        if any(w in q_norm for w in ["stampac", "stampaci", "printer", "oprema", "ploter"]):
            queries.append("HP Designjet Canon TX Kyocera ploter štampač oprema toneri")

        # Detekcija člana (i na ćirilici i na latinici)
        article_match = re.search(r'(?:clan|član|члан|cl|čl|чл)\.?\s*(\d+)', q_norm)
        if article_match:
            art_num = article_match.group(1)
            queries.append(f"Kolektivni ugovor Član {art_num}")
            queries.append(f"Колективни уговор Члан {art_num}")
            queries.append(f"Član {art_num}. clan {art_num} čl {art_num}")
            queries.append(f"Члан {art_num} чл {art_num}")

        return queries

    def query(self, user_question: str) -> dict:
        context_texts = []
        sources = []
        
        q_norm = normalize_text(user_question)
        is_internal_query = any(w in q_norm for w in INTERNAL_WORDS)
        
        article_match = re.search(r'(?:clan|član|члан|cl|čl|чл)\.?\s*(\d+)', q_norm)
        target_article = article_match.group(1) if article_match else None

        # 1. PRETRAGA LOKALNE DOKUMENTACIJE (QDRANT)
        try:
            search_queries = self._generate_search_queries(user_question)
            all_retrieved_docs = []
            seen_contents = set()

            for q_str in search_queries:
                docs = self.vector_store.similarity_search(q_str, k=15)
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

            # HIBRIDNO RE-RANGIRANJE ZA ČLANOVE (Uključuje i Ćirilicu)
            if target_article:
                def score_chunk(chunk):
                    c_norm = normalize_text(chunk["text"])
                    # Provera prisustva broja člana i ključnih reči za ugovor
                    has_num = re.search(r'\b' + re.escape(target_article) + r'\b', c_norm)
                    has_clan = any(w in c_norm for w in ["clan", "cl", "claba"])
                    has_ugovor = "kolektivn" in c_norm or "ugovor" in c_norm
                    
                    if has_num and (has_clan or has_ugovor):
                        if re.search(r'\b(?:clan|cl)\.?\s*' + re.escape(target_article) + r'\b', c_norm):
                            return 200
                        return 100
                    return 0

                extracted_chunks.sort(key=score_chunk, reverse=True)

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
            
        # 2. PRETRAGA WEBA (TAVILY API)
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
            "STRUKTURA ODGOVORA:\n"
            "1. KADA JE PITANJE 'Koji štampači se koriste u Birou?' ili opšte o štampačima/opremi:\n"
            "   - Navedi SAMO nazive modela uređaja (HP Designjet 800PS, Canon TX-3000, Kyocera FS-9530dn, Kyocera M3655idn, Kyocera P2040dn).\n"
            "   - NIKADA nemoj navoditi šifre tonera i količine ako se pitaju samo štampači.\n"
            "2. KADA SE TRAŽI KONKRETAN ČLAN (npr. Član 14 Kolektivnog ugovora):\n"
            "   - Pronađi taj član u kontekstu i citiraj ili detaljno sumiraj sve njegove odredbe.\n"
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