import os
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

    def query(self, user_question: str) -> dict:
        try:
            # Smanjeno k=3 radi čuvanja tokena u okviru Groq 6000 TPM limita
            docs = self.vector_store.similarity_search(user_question, k=3)
        except Exception as e:
            return {"answer": f"⚠️ Greška prilikom pretrage baze: {e}", "source_documents": []}
            
        if not docs:
            return {"answer": "Nažalost, nisam pronašao relevantne informacije u bazi podataka.", "source_documents": []}
            
        context_texts = []
        sources = []
        
        for doc in docs:
            text_content = doc.page_content.strip()
            # Skraćivanje pojedinačnih odsečaka ako su predugački
            if len(text_content) > 800:
                text_content = text_content[:800] + "..."
                
            source_file = doc.metadata.get("source", "Nepoznat dokument")
            if source_file not in sources:
                sources.append(source_file)
            
            block = f"Izvor: {source_file}\n{text_content}"
            context_texts.append(block)
            
        context = "\n\n---\n\n".join(context_texts)
        
        # Tvrdo ograničenje konteksta na max 3500 karaktera (~850 tokena)
        if len(context) > 3500:
            context = context[:3500] + "\n...[Kontekst skraćen radi limita]"
        
        system_prompt = (
            "Ti si BiroChat, korporativni asistent za pretragu dokumentacije. "
            "Odgovori kratko i precizno na pitanje koristeći ISKLJUČIVO navedeni kontekst. "
            "Ako u kontekstu nema traženog podatka, napiši da podatak nije direktno naveden u dokumentima."
        )
        
        user_prompt = f"Kontekst iz baze:\n{context}\n\nPitanje: {user_question}"
        
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
            return {"answer": f"⚠️ Greška prilikom generisanja odgovora: {e}", "source_documents": []}

_engine_instance = RAGEngine()

def ask_birochat(question: str) -> dict:
    return _engine_instance.query(question)