"""
biro_chain.py — LangChain-based Multimodal RAG
==============================================
Srce novog sistema. Koristi:
- LangChain za orkestraciju
- fastembed za text embeddings (free, local)
- CLIP za image embeddings (free, local)
- Qdrant za vector store
- Groq za LLM (free, 30 req/min)

Moduli:
- BiroEmbeddings: text + image embedding
- BiroRetriever: Multi-Vector retriever (text + image summaries)
- BiroChain: LLM chain sa context-om

Konfiguracija env varijabli:
    QDRANT_URL, QDRANT_API_KEY
    GROQ_API_KEY
    COLLECTION_NAME (default: baza_biro_v2)
    GEMINI_API_KEY (za Vision analizu)
"""

import os
import base64
from typing import List, Optional, Tuple
from pathlib import Path

# LangChain
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    UnstructuredMarkdownLoader,
)

# Qdrant
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct,
    Filter, FieldCondition, MatchValue,
)

# Fastembed (free, local)
try:
    from fastembed import TextEmbedding, ImageEmbedding
    FASTEMBED_AVAILABLE = True
except ImportError:
    FASTEMBED_AVAILABLE = False

# Groq
from langchain_groq import ChatGroq


# ============================================================
# EMBEDDINGS
# ============================================================
class BiroEmbeddings(Embeddings):
    """Multimodal embeddings: text + image.

    Text: intfloat/multilingual-e5-large (1024 dim) — podržava ćirilicu, latinicu, 100+ jezika
    Image: CLIP-ViT-B-32 (512 dim)
    """

    def __init__(self, model_name: str = "intfloat/multilingual-e5-large"):
        if not FASTEMBED_AVAILABLE:
            raise ImportError("pip install fastembed")

        # Pokušaj više multilingual modela
        for candidate in [
            model_name,
            "intfloat/multilingual-e5-large",
            "intfloat/multilingual-e5-base",
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        ]:
            try:
                self.text_model = TextEmbedding(model_name=candidate)
                # Dinamički odredi dim
                test_vec = list(self.text_model.embed(["test"]))[0]
                self.text_dim = len(test_vec.tolist())
                self.text_model_name = candidate
                print(f"   ✅ Text embedding: {candidate} ({self.text_dim}d)")
                break
            except Exception as e:
                print(f"   ⚠ {candidate} ne radi: {str(e)[:60]}")
                continue
        else:
            raise ValueError("Nijedan multilingual model ne radi")

        # Image embedding (CLIP) — probaj više modela
        for img_model in [
            "Qdrant/clip-ViT-B-32-vision",
            "Qdrant/Unicom-ViT-B-32",
            "jinaai/jina-clip-v1",
        ]:
            try:
                self.image_model = ImageEmbedding(model_name=img_model)
                test_vec = list(self.image_model.embed(["test_dummy_path"]))[0] if False else None
                # Skip test, just init
                self.image_model_name = img_model
                print(f"   ✅ Image embedding: {img_model}")
                break
            except Exception as e:
                print(f"   ⚠ {img_model} ne radi: {str(e)[:60]}")
                continue
        else:
            print("   ⚠ Nijedan image model ne radi — image search neće biti dostupan")
            self.image_model = None
        self.image_dim = 512

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed list of texts."""
        embeddings = list(self.text_model.embed(texts))
        return [e.tolist() for e in embeddings]

    def embed_query(self, text: str) -> List[float]:
        """Embed single query."""
        embeddings = list(self.text_model.embed([text]))
        return embeddings[0].tolist()

    def embed_image(self, image_path_or_url: str) -> Optional[List[float]]:
        """Embed image from path or URL."""
        try:
            if image_path_or_url.startswith("http"):
                # Download image
                import requests
                r = requests.get(image_path_or_url, timeout=10)
                if r.status_code != 200:
                    return None
                # Save to temp
                tmp_path = Path("/tmp") / Path(image_path_or_url).name
                with open(tmp_path, "wb") as f:
                    f.write(r.content)
                image_path = str(tmp_path)
            else:
                image_path = image_path_or_url

            embeddings = list(self.image_model.embed([image_path]))
            return embeddings[0].tolist()
        except Exception as e:
            print(f"Image embed error: {e}")
            return None


# ============================================================
# DOCUMENT LOADERS
# ============================================================
def load_document(file_path: str, ocr=None, ocr_engine: str = "auto") -> List[Document]:
    """Učitava dokument i vraća listu Document objekata.

    Podržava: PDF, DOCX, DOC, MD, TXT, CSV, XLSX, slike (PNG, JPG, ...)
    Ako je slika i prosleđen je OCR, izvlači tekst iz slike.
    """
    from langchain_community.document_loaders import TextLoader
    path = Path(file_path)
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        loader = PyPDFLoader(file_path)
    elif suffix in [".docx", ".doc"]:
        loader = Docx2txtLoader(file_path)
    elif suffix in [".md", ".markdown", ".txt"]:
        loader = TextLoader(file_path, encoding="utf-8")
    elif suffix == ".csv":
        # CSV kao tekst
        loader = TextLoader(file_path, encoding="utf-8")
    elif suffix in [".xlsx", ".xls"]:
        # XLSX — koristimo unstructured ili pandas
        try:
            import pandas as pd
            df = pd.read_excel(file_path)
            text = df.to_string()
            return [Document(
                page_content=text,
                metadata={"source": file_path, "type": "xlsx", "filename": path.name},
            )]
        except Exception as e:
            return [Document(
                page_content=f"[XLSX greška: {e}]",
                metadata={"source": file_path, "type": "xlsx", "filename": path.name, "error": str(e)},
            )]
    elif suffix in [".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"]:
        return _load_image(path, ocr, ocr_engine)
    else:
        raise ValueError(f"Nepoznat format: {suffix}")

    return loader.load()


def _load_image(path: Path, ocr=None, ocr_engine: str = "auto") -> List[Document]:
    """Učitava sliku i izvlači tekst preko OCR-a ako je dostupan.

    Podržava: PaddleOCR, EasyOCR
    Workaround za Unicode putanje: kopiraj u privremeni ASCII fajl.
    """
    if ocr is None:
        return [Document(
            page_content=f"[Slika: {path.name}]",
            metadata={"source": str(path), "type": "image", "filename": path.name},
        )]

    # Workaround za Unicode putanje (OpenCV ima problem sa njima)
    import shutil
    import tempfile
    tmp_path = None
    try:
        if not path.name.isascii():
            # Napravi privremeni ASCII fajl
            tmp_dir = Path(tempfile.gettempdir())
            tmp_path = tmp_dir / f"ocr_temp_{hash(str(path)) & 0xFFFFFFFF}.jpg"
            shutil.copy(str(path), str(tmp_path))
            ocr_input = str(tmp_path)
        else:
            ocr_input = str(path)
    except Exception:
        ocr_input = str(path)

    try:
        # EasyOCR format: list of (bbox, text, conf)
        if ocr_engine == "easyocr":
            try:
                result = ocr.readtext(ocr_input)
                text_lines = [line[1] for line in result if len(line) >= 2]
                text = "\n".join(text_lines)
                return [Document(
                    page_content=text or f"[Slika: {path.name}, OCR prazan]",
                    metadata={
                        "source": str(path),
                        "type": "image",
                        "filename": path.name,
                        "ocr_engine": "easyocr",
                        "ocr_lines": len(text_lines),
                    },
                )]
            except Exception as e:
                return [Document(
                    page_content=f"[Slika: {path.name}, EasyOCR greška: {e}]",
                    metadata={"source": str(path), "type": "image", "filename": path.name, "error": str(e)},
                )]

        # PaddleOCR
        text_lines = []
        try:
            try:
                result = ocr.predict(ocr_input)
            except AttributeError:
                result = ocr.ocr(ocr_input, cls=True)

            if result:
                for page in result:
                    if hasattr(page, 'get'):
                        rec_texts = page.get('rec_texts', [])
                        text_lines.extend(rec_texts)
                    elif isinstance(page, list):
                        for line in page:
                            if line and len(line) >= 2:
                                text_lines.append(line[1][0] if isinstance(line[1], tuple) else line[1])

            text = "\n".join(text_lines)
            return [Document(
                page_content=text or f"[Slika: {path.name}, OCR prazan]",
                metadata={
                    "source": str(path),
                    "type": "image",
                    "filename": path.name,
                    "ocr_engine": "paddleocr",
                    "ocr_lines": len(text_lines),
                },
            )]
        except Exception as e:
            return [Document(
                page_content=f"[Slika: {path.name}, OCR greška: {e}]",
                metadata={"source": str(path), "type": "image", "filename": path.name, "error": str(e)},
            )]
    finally:
        # Cleanup privremenog fajla
        if tmp_path and tmp_path.exists():
            try:
                tmp_path.unlink()
            except Exception:
                pass


def split_documents(
    documents: List[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> List[Document]:
    """Splits documents into smaller chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(documents)


# ============================================================
# RETRIEVER
# ============================================================
class BiroRetriever(BaseRetriever):
    """Multi-Vector Retriever za Biro.

    Tri kolekcije u Qdrant:
    - text_chunks: text + vizuel_opis embeddings (768 dim)
    - images: image embeddings (512 dim)
    - metadata: filterable polja
    """

    qdrant: QdrantClient
    embeddings: BiroEmbeddings
    collection_text: str = "biro_text"
    collection_images: str = "biro_images"
    k: int = 5

    class Config:
        arbitrary_types_allowed = True

    def _get_relevant_documents(
        self, query: str, *, run_manager: Optional[CallbackManagerForRetrieverRun] = None
    ) -> List[Document]:
        """Retrieve relevant documents for a query."""
        query_vec = self.embeddings.embed_query(query)

        # 1. Text retrieval
        try:
            text_results = self.qdrant.query_points(
                collection_name=self.collection_text,
                query=query_vec,
                limit=self.k,
            )
            text_docs = []
            for point in text_results.points:
                payload = point.payload or {}
                text_docs.append(Document(
                    page_content=payload.get("text", ""),
                    metadata=payload,
                ))
        except Exception:
            text_docs = []

        # 2. Image retrieval (semantic)
        # TODO: query za image embeddings
        # Za sada samo text
        return text_docs


# ============================================================
# CHAIN
# ============================================================
def create_llm(model: str = "llama-3.1-8b-instant"):
    """Kreira Groq LLM."""
    return ChatGroq(
        model=model,
        temperature=0.1,
        max_tokens=500,
        api_key=os.environ.get("GROQ_API_KEY"),
    )


def create_chain(retriever: BiroRetriever, llm=None):
    """Kreira RetrievalQA chain."""
    from langchain.chains import RetrievalQA
    from langchain.prompts import PromptTemplate

    if llm is None:
        llm = create_llm()

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""Ti si asistent za PD Srbijašume, Biro za planiranje.
Odgovori na pitanje koristeći SAMO dati kontekst. Ako nema dovoljno informacija, reci "Nemam dovoljno informacija".

KONTEKST:
{context}

PITANJE: {question}

ODGOVOR:""",
    )

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True,
    )
    return chain


# ============================================================
# UTILITY
# ============================================================
def get_qdrant_client():
    """Vraća Qdrant klijent."""
    url = os.environ.get("QDRANT_URL")
    api_key = os.environ.get("QDRANT_API_KEY")

    if not url:
        print("❌ QDRANT_URL env varijabla nije postavljena!")
        print("   Postavi: $env:QDRANT_URL='https://tvoj-cluster.cloud.qdrant.io'")
        raise ValueError("QDRANT_URL missing")
    if not api_key:
        print("❌ QDRANT_API_KEY env varijabla nije postavljena!")
        print("   Postavi: $env:QDRANT_API_KEY='tvoj_api_kljuc'")
        raise ValueError("QDRANT_API_KEY missing")

    print(f"   🔗 Qdrant: {url[:50]}...")
    return QdrantClient(
        url=url,
        api_key=api_key,
        check_compatibility=False,
        timeout=30,
    )


def ensure_collections(qdrant: QdrantClient, collection_prefix: str = "biro", text_dim: int = 1024):
    """Kreira kolekcije ako ne postoje, sa payload indeksima za filtere."""
    collections = [c.name for c in qdrant.get_collections().collections]

    # Text collection
    text_col = f"{collection_prefix}_text"
    if text_col not in collections:
        qdrant.create_collection(
            collection_name=text_col,
            vectors_config=VectorParams(size=text_dim, distance=Distance.COSINE),
        )
        # Payload indeksi za filtere
        try:
            qdrant.create_payload_index(
                collection_name=text_col,
                field_name="filename",
                field_schema="keyword",
            )
            qdrant.create_payload_index(
                collection_name=text_col,
                field_name="source",
                field_schema="keyword",
            )
        except Exception as e:
            print(f"   ⚠ Index creation: {e}")

    # Image collection (512 dim)
    image_col = f"{collection_prefix}_images"
    if image_col not in collections:
        qdrant.create_collection(
            collection_name=image_col,
            vectors_config=VectorParams(size=512, distance=Distance.COSINE),
        )
        try:
            qdrant.create_payload_index(
                collection_name=image_col,
                field_name="filename",
                field_schema="keyword",
            )
        except Exception:
            pass

    return text_col, image_col


def list_available_models():
    """Pomoćna funkcija — lista sve dostupne modele."""
    if not FASTEMBED_AVAILABLE:
        print("fastembed nije instaliran")
        return
    print("TextEmbedding podržani modeli:")
    try:
        for m in TextEmbedding.list_supported_models():
            print(f"  - {m.get('model', m)}")
    except Exception as e:
        print(f"Greška: {e}")
    print("\nImageEmbedding podržani modeli:")
    try:
        for m in ImageEmbedding.list_supported_models():
            print(f"  - {m.get('model', m)}")
    except Exception as e:
        print(f"Greška: {e}")
