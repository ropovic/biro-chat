"""
biro_ingest.py — LangChain document ingestion sa PaddleOCR
============================================================
Ingest dokumenata (PDF, DOCX, MD, slike) u Qdrant bazu.

Pipeline:
1. Load dokument (PyPDFLoader, Docx2txtLoader, ...)
2. Split u chunkove (RecursiveCharacterTextSplitter)
3. (Opciono) PaddleOCR za skenirane PDF-ove
4. Embed chunks (fastembed text embeddings)
5. Store u Qdrant

PaddleOCR instalacija:
    pip install paddlepaddle paddleocr
    # Prvo pokretanje download-uje model (~100MB)

Pokretanje:
    # Pojedinačni fajl
    python biro_ingest.py ingest file.pdf

    # Ceo folder
    python biro_ingest.py ingest /path/to/docs/

    # Bez OCR-a
    python biro_ingest.py ingest folder/ --no-ocr

    # Samo statistika
    python biro_ingest.py stats
"""

import os
import sys
import time
import json
import argparse
import hashlib
from pathlib import Path
from typing import List, Optional

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    UnstructuredMarkdownLoader,
    TextLoader,
)
from qdrant_client import QdrantClient
from qdrant_client import models
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue

# Naš LangChain modul
from biro_chain import (
    BiroEmbeddings,
    get_qdrant_client,
    ensure_collections,
    load_document,
    split_documents,
)

# PaddleOCR (opciono)
PADDLEOCR_AVAILABLE = False
try:
    from paddleocr import PaddleOCR
    PADDLEOCR_AVAILABLE = True
except ImportError:
    pass

# EasyOCR (backup)
EASYOCR_AVAILABLE = False
try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    pass


# ============================================================
# INGESTOR
# ============================================================
class BiroIngestor:
    """Glavna klasa za ingestion dokumenata."""

    SUPPORTED_EXTENSIONS = {
        ".pdf", ".docx", ".doc", ".md", ".markdown", ".txt",
        # Tabele
        ".csv", ".xlsx", ".xls",
        # Slike (treba PaddleOCR/EasyOCR)
        ".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"
    }

    def __init__(
        self,
        collection_prefix: str = "biro",
        use_paddleocr: bool = False,
        ocr_engine: str = "auto",
        use_gpu: bool = False,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ):
        print("🔧 Inicijalizujem BiroIngestor...")
        self.use_gpu = use_gpu

        # Embeddings
        self.embeddings = BiroEmbeddings()

        # Qdrant
        self.qdrant = get_qdrant_client()
        self.text_col, self.image_col = ensure_collections(
            self.qdrant, collection_prefix
        )
        self.collection_prefix = collection_prefix

        # PaddleOCR ili EasyOCR
        self.ocr = None
        self.ocr_engine_name = None

        # Detaljno biranje OCR-a
        if use_paddleocr or ocr_engine in ("paddleocr", "auto"):
            if PADDLEOCR_AVAILABLE:
                print("   📷 Učitavam PaddleOCR (prvi put ~30s)...")
                for lang in ["hr", "bs", "en", None]:
                    try:
                        kwargs = {"use_textline_orientation": True}
                        if lang:
                            kwargs["lang"] = lang
                        self.ocr = PaddleOCR(**kwargs)
                        self.ocr_engine_name = "paddleocr"
                        print(f"   ✅ PaddleOCR init OK (lang={lang or 'multi'})")
                        # Test da li STVARNO radi
                        try:
                            test = self.ocr.predict("test")
                        except Exception as test_err:
                            print(f"   ⚠ PaddleOCR init ali predict ne radi: {str(test_err)[:60]}")
                            self.ocr = None
                            continue
                        break
                    except Exception as e:
                        print(f"   ⚠ PaddleOCR {lang} ne radi: {str(e)[:50]}")
                        continue

        # Ako PaddleOCR ne radi, probaj EasyOCR
        if (self.ocr is None or ocr_engine == "easyocr") and EASYOCR_AVAILABLE:
            if ocr_engine == "easyocr" or use_paddleocr or ocr_engine == "auto":
                print("   📷 Učitavam EasyOCR...")
                try:
                    import easyocr
                    # Proveri da li je GPU dostupan
                    gpu_available = False
                    if self.use_gpu:
                        try:
                            import torch
                            gpu_available = torch.cuda.is_available()
                            if gpu_available:
                                gpu_name = torch.cuda.get_device_name(0)
                                print(f"   🎮 GPU: {gpu_name}")
                            else:
                                print("   ⚠ PyTorch nema CUDA, koristim CPU")
                        except ImportError:
                            print("   ⚠ PyTorch nije instaliran, koristim CPU")
                    self.ocr = easyocr.Reader(['hr', 'en'], gpu=gpu_available)
                    self.ocr_engine_name = "easyocr"
                    mode = "GPU" if gpu_available else "CPU"
                    print(f"   ✅ EasyOCR radi ({mode})")
                except Exception as e:
                    print(f"   ❌ EasyOCR ne radi: {e}")
        elif use_paddleocr and not PADDLEOCR_AVAILABLE and not EASYOCR_AVAILABLE:
            print("   ⚠ OCR nije dostupan")
            print("   pip install paddleocr  # ili  pip install easyocr")

        # Text splitter
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", "? ", "! ", " ", ""],
        )

        print(f"   ✅ Spremno. Kolekcije: {self.text_col}, {self.image_col}")

    def _make_id(self, content: str) -> str:
        """Generiše stabilan UUID iz sadržaja (Qdrant zahteva UUID)."""
        import uuid
        # Koristimo UUID5 sa namespace-om — deterministički iz content-a
        return str(uuid.uuid5(uuid.NAMESPACE_DNS, content))

    def ingest_file(self, file_path: str, verbose: bool = True) -> dict:
        """Ingest jedan fajl. Vraća statistiku."""
        path = Path(file_path)
        if not path.exists():
            return {"error": f"Fajl ne postoji: {file_path}"}

        suffix = path.suffix.lower()
        if suffix not in self.SUPPORTED_EXTENSIONS:
            return {"error": f"Nepodržan format: {suffix}"}

        if verbose:
            print(f"\n📄 {path.name} ({suffix})")

        start = time.time()

        try:
            # 1. Load (prosleđujemo OCR ako postoji)
            docs = load_document(str(path), ocr=self.ocr, ocr_engine=self.ocr_engine_name)
            if verbose:
                print(f"   Loaded: {len(docs)} pages/sections")
                if docs and docs[0].page_content:
                    preview = docs[0].page_content[:200].replace("\n", " ")
                    print(f"   Preview: {preview}...")

            # 2. Split
            chunks = self.splitter.split_documents(docs)
            if verbose:
                print(f"   Split: {len(chunks)} chunks")

            if not chunks:
                return {"file": path.name, "chunks": 0}

            # 3. Embed + Store
            texts = [c.page_content for c in chunks]
            vectors = self.embeddings.embed_documents(texts)

            # 4. Kreiraj PointStruct sa UUID
            points = []
            for i, (chunk, vector) in enumerate(zip(chunks, vectors)):
                # UUID5 je deterministički — isti sadržaj uvek daje isti ID
                point_id = self._make_id(f"{path.name}::{i}::{chunk.page_content[:100]}")
                payload = {
                    "text": chunk.page_content,
                    "source": str(path),
                    "filename": path.name,
                    "chunk_index": i,
                    "metadata": chunk.metadata,
                }
                points.append(PointStruct(
                    id=point_id,
                    vector=vector,
                    payload=payload,
                ))

            # 5. Upsert u Qdrant
            self.qdrant.upsert(
                collection_name=self.text_col,
                points=points,
            )

            elapsed = time.time() - start
            if verbose:
                print(f"   ✅ Stored: {len(points)} chunks ({elapsed:.1f}s)")

            return {
                "file": path.name,
                "chunks": len(points),
                "time": elapsed,
            }

        except Exception as e:
            return {"file": path.name, "error": str(e)}

    def ingest_folder(
        self,
        folder_path: str,
        recursive: bool = True,
        skip_existing: bool = True,
    ) -> dict:
        """Ingest sve fajlove iz foldera."""
        folder = Path(folder_path)
        if not folder.exists():
            return {"error": f"Folder ne postoji: {folder_path}"}

        # Nađi sve podržane fajlove
        if recursive:
            files = []
            for ext in self.SUPPORTED_EXTENSIONS:
                files.extend(folder.rglob(f"*{ext}"))
        else:
            files = []
            for ext in self.SUPPORTED_EXTENSIONS:
                files.extend(folder.glob(f"*{ext}"))

        if not files:
            return {"error": "Nema podržanih fajlova"}

        print(f"\n📁 Pronađeno {len(files)} fajlova u {folder}")

        results = {"success": 0, "failed": 0, "skipped": 0, "details": []}

        for i, file_path in enumerate(files):
            print(f"\n[{i+1}/{len(files)}]", end=" ")

            # Proveri da li već postoji (po imenu fajla)
            if skip_existing:
                try:
                    existing = self.qdrant.scroll(
                        collection_name=self.text_col,
                        scroll_filter=models.Filter(
                            must=[
                                models.FieldCondition(
                                    key="filename",
                                    match=models.MatchValue(value=file_path.name),
                                )
                            ]
                        ),
                        limit=1,
                    )
                    if existing[0]:
                        print(f"⏭ {file_path.name} (already ingested)")
                        results["skipped"] += 1
                        continue
                except Exception as e:
                    # Ako filter ne radi, probaj Python-side
                    all_records, _ = self.qdrant.scroll(
                        collection_name=self.text_col,
                        limit=10000,
                        with_payload=True,
                        with_vectors=False,
                    )
                    filenames = {r.payload.get("filename", "") for r in all_records}
                    if file_path.name in filenames:
                        print(f"⏭ {file_path.name} (already ingested)")
                        results["skipped"] += 1
                        continue

            result = self.ingest_file(str(file_path))
            results["details"].append(result)

            if "error" in result:
                results["failed"] += 1
                # PRIKAŽI GREŠKU!
                print(f"   ❌ GREŠKA: {result.get('error', 'nepoznata')[:200]}")
            else:
                results["success"] += 1

        return results

    def stats(self) -> dict:
        """Vraća statistiku baze."""
        text_info = self.qdrant.get_collection(self.text_col)
        image_info = self.qdrant.get_collection(self.image_col)

        # Qdrant client API se menja između verzija — robustan pristup
        def safe_count(info, key):
            # Probaj više atributa
            for attr in [key, f"{key}_count", f"total_{key}_count"]:
                val = getattr(info, attr, None)
                if val is not None:
                    return val
            # Ili config.params.vectors.size
            config = getattr(info, "config", None)
            if config:
                params = getattr(config, "params", None)
                if params:
                    vectors = getattr(params, "vectors", None)
                    if isinstance(vectors, dict):
                        return vectors.get("size", "?")
            return "?"

        return {
            "text_collection": {
                "name": self.text_col,
                "vectors": safe_count(text_info, "vectors"),
                "points": safe_count(text_info, "points"),
            },
            "image_collection": {
                "name": self.image_col,
                "vectors": safe_count(image_info, "vectors"),
                "points": safe_count(image_info, "points"),
            },
        }

    def clear_collection(self, collection: str = "text"):
        """Briše sve iz kolekcije."""
        target = self.text_col if collection == "text" else self.image_col
        self.qdrant.delete_collection(target)
        print(f"🗑️  Kolekcija {target} obrisana")
        # Rekreiraj
        ensure_collections(self.qdrant, self.collection_prefix)


# ============================================================
# CLI
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="Biro Ingest — LangChain document ingestion"
    )
    sub = parser.add_subparsers(dest="command")

    # ingest
    p_ingest = sub.add_parser("ingest", help="Ingest fajl ili folder")
    p_ingest.add_argument("path", help="Putanja do fajla ili foldera")
    p_ingest.add_argument("--no-recursive", action="store_true")
    p_ingest.add_argument("--force", action="store_true",
                          help="Ingiraj i ako već postoji")
    p_ingest.add_argument("--use-paddleocr", action="store_true",
                          help="Koristi PaddleOCR za slike")
    p_ingest.add_argument("--use-easyocr", action="store_true",
                          help="Koristi EasyOCR za slike (stabilnije od PaddleOCR)")
    p_ingest.add_argument("--ocr-engine", choices=["auto", "paddleocr", "easyocr"],
                          default="auto", help="Koji OCR koristiti (default: auto)")
    p_ingest.add_argument("--use-gpu", action="store_true",
                          help="Koristi GPU za OCR (zahteva CUDA PyTorch)")
    p_ingest.add_argument("--collection-prefix", type=str, default="biro",
                          help="Prefix za Qdrant kolekcije (default: biro)")
    p_ingest.add_argument("--skip-pattern", type=str, action="append", default=[],
                          help="Preskoči fajlove koji sadrže ovaj pattern (npr. 'Photo')")

    # stats
    sub.add_parser("stats", help="Prikaži statistiku")

    # clear
    p_clear = sub.add_parser("clear", help="Obriši kolekciju")
    p_clear.add_argument("collection", choices=["text", "images", "both"])

    # search (test)
    p_search = sub.add_parser("search", help="Test pretraga")
    p_search.add_argument("query")
    p_search.add_argument("--k", type=int, default=5)

    args = parser.parse_args()

    if args.command == "ingest":
        use_paddle = args.use_paddleocr or args.ocr_engine in ("paddleocr", "auto")
        use_easy = args.use_easyocr or args.ocr_engine == "easyocr"
        # Ako je eksplicitno easyocr, forsiraj
        if args.ocr_engine == "easyocr":
            use_paddle = False
        ingestor = BiroIngestor(
            use_paddleocr=use_paddle,
            ocr_engine=args.ocr_engine,
            use_gpu=args.use_gpu,
            collection_prefix=args.collection_prefix,
        )

        # Custom folder() sa skip patterns
        path = Path(args.path)
        if path.is_file():
            result = ingestor.ingest_file(str(path))
            print(f"\n📊 {result}")
            if "error" in result:
                print(f"   ❌ Greška: {result['error'][:200]}")
        elif path.is_dir():
            # Modifikuj ingest_folder da podrži skip_pattern
            from qdrant_client import models as _models

            # Nađi sve fajlove
            files = []
            for ext in ingestor.SUPPORTED_EXTENSIONS:
                if args.no_recursive:
                    files.extend(path.glob(f"*{ext}"))
                else:
                    files.extend(path.rglob(f"*{ext}"))

            # Filtriraj po skip patterns
            if args.skip_pattern:
                original = len(files)
                files = [f for f in files if not any(p in str(f) for p in args.skip_pattern)]
                print(f"⏭  Skip patterns: {args.skip_pattern} ({original - len(files)} fajlova preskočeno)")

            if not files:
                print("Nema fajlova za obradu")
                return

            print(f"\n📁 Pronađeno {len(files)} fajlova")

            results = {"success": 0, "failed": 0, "skipped": 0, "details": []}

            for i, file_path in enumerate(files):
                print(f"\n[{i+1}/{len(files)}]", end=" ")

                # Provera da li već postoji
                if not args.force:
                    try:
                        existing = ingestor.qdrant.scroll(
                            collection_name=ingestor.text_col,
                            scroll_filter=_models.Filter(
                                must=[
                                    _models.FieldCondition(
                                        key="filename",
                                        match=_models.MatchValue(value=file_path.name),
                                    )
                                ]
                            ),
                            limit=1,
                        )
                        if existing[0]:
                            print(f"⏭ {file_path.name}")
                            results["skipped"] += 1
                            continue
                    except Exception:
                        pass

                result = ingestor.ingest_file(str(file_path))
                results["details"].append(result)

                if "error" in result:
                    results["failed"] += 1
                    print(f"   ❌ {result['error'][:150]}")
                else:
                    results["success"] += 1
                    if (i + 1) % 10 == 0:
                        print(f"\n💾 Progress: {results['success']}/{i+1}")

            print(f"\n{'='*60}")
            print(f"📊 GOTOVO")
            print(f"   Success: {results['success']}")
            print(f"   Failed:  {results['failed']}")
            print(f"   Skipped: {results['skipped']}")
        else:
            print(f"❌ Putanja ne postoji: {args.path}")

    elif args.command == "stats":
        ingestor = BiroIngestor()
        stats = ingestor.stats()
        print(f"\n📊 Statistika baze:")
        for k, v in stats.items():
            print(f"   {k}: {v['points']} points ({v['vectors']} vectors)")

    elif args.command == "clear":
        ingestor = BiroIngestor()
        if args.collection in ["text", "both"]:
            ingestor.clear_collection("text")
        if args.collection in ["images", "both"]:
            ingestor.clear_collection("images")

    elif args.command == "search":
        ingestor = BiroIngestor()
        vec = ingestor.embeddings.embed_query(args.query)
        results = ingestor.qdrant.query_points(
            collection_name=ingestor.text_col,
            query=vec,
            limit=args.k,
        )
        print(f"\n🔍 Query: {args.query}")
        print(f"   Vektor dim: {len(vec)}")
        print(f"\n📋 Rezultati:")
        for i, p in enumerate(results.points):
            text = p.payload.get("text", "")[:150]
            print(f"\n[{i+1}] Score: {p.score:.3f}")
            print(f"    {text}...")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
