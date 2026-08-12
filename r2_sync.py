import os
import json
import boto3
import streamlit as st
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
from sentence_transformers import SentenceTransformer


# --- BEZBEDNO ČITANJE PODEŠAVANJA (BEZ CONFIG.PY) ---
def get_config(key: str, default: str = "") -> str:
    try:
        if key in st.secrets:
            return str(st.secrets[key])
    except Exception:
        pass
    return os.getenv(key, default)


R2_ACCOUNT_ID = get_config("R2_ACCOUNT_ID")
R2_ACCESS_KEY_ID = get_config("R2_ACCESS_KEY_ID")
R2_SECRET_ACCESS_KEY = get_config("R2_SECRET_ACCESS_KEY")
R2_BUCKET_NAME = get_config("R2_BUCKET_NAME")
R2_PUBLIC_URL_PREFIX = get_config("R2_PUBLIC_URL_PREFIX").rstrip("/")

QDRANT_URL = get_config("QDRANT_URL", "https://09ffa5ef-2765-45c8-bfcf-29bc6bf90f08.eu-west-2-0.aws.cloud.qdrant.io")
QDRANT_API_KEY = get_config("QDRANT_API_KEY", "")
COLLECTION_NAME = get_config("COLLECTION_NAME", "biro_documents")


def get_s3_client():
    if not R2_ACCOUNT_ID or not R2_ACCESS_KEY_ID or not R2_SECRET_ACCESS_KEY:
        raise ValueError("Cloudflare R2 kredencijali nisu podešeni u Secrets.")
        
    return boto3.client(
        service_name="s3",
        endpoint_url=f"https://{R2_ACCOUNT_ID}.r2.cloudflarestorage.com",
        aws_access_key_id=R2_ACCESS_KEY_ID,
        aws_secret_access_key=R2_SECRET_ACCESS_KEY,
        region_name="auto"
    )


def sync_employees_from_r2():
    """
    Preuzima 'employees.json' iz Cloudflare R2 bucket-a i indeksira
    sve zaposlene, rukovodstvo i logotipe u Qdrant vektorsku bazu.
    """
    print("Preuzimanje manifesta 'employees.json' iz Cloudflare R2...")
    s3 = get_s3_client()

    try:
        response = s3.get_object(Bucket=R2_BUCKET_NAME, Key="employees.json")
        raw_data = response['Body'].read().decode('utf-8')
        employees_data = json.loads(raw_data)
    except Exception as e:
        print(f"❌ Greška pri čitanju employees.json iz R2 bucket-a: {e}")
        return

    print(f"Pronađeno {len(employees_data)} unosa. Inicijalizacija BAAI/bge-m3 embedding modela...")
    embedder = SentenceTransformer("BAAI/bge-m3")
    qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, check_compatibility=False)

    points = []
    for idx, emp in enumerate(employees_data):
        doc_type = emp.get("doc_type", "employee")
        photo_filename = emp.get("photo_filename", "")
        
        # Formiranje javnog URL-a fotografije iz R2 bucket-a
        if R2_PUBLIC_URL_PREFIX:
            image_url = f"{R2_PUBLIC_URL_PREFIX}/{photo_filename}"
        else:
            image_url = photo_filename

        # Priprema teksta za vektorsku pretragu
        full_text = (
            f"Ime i prezime / Naziv: {emp.get('name', '')}. "
            f"Rola / Funkcija: {emp.get('role', '')}. "
            f"Tip unosa: {doc_type}. "
            f"Da li je direktor: {'Da' if emp.get('is_director') else 'Ne'}. "
            f"Da li je zamenik direktora: {'Da' if emp.get('is_deputy_director') else 'Ne'}. "
            f"Detaljan opis: {emp.get('description', '')}"
        )

        vector = embedder.encode(full_text).tolist()

        payload = {
            "doc_type": doc_type,
            "name": emp.get("name", ""),
            "role": emp.get("role", "Zaposleni"),
            "is_director": emp.get("is_director", False),
            "is_deputy_director": emp.get("is_deputy_director", False),
            "description": emp.get("description", ""),
            "photo_filename": photo_filename,
            "image_url": image_url,
            "text": full_text
        }

        points.append(
            PointStruct(
                id=20000 + idx,  # Unikatni numerički opseg ID-jeva za zaposlene
                vector=vector,
                payload=payload
            )
        )

    # Provera i kreiranje Qdrant kolekcije sa vektorskom dimenzijom 1024 za BAAI/bge-m3
    existing_collections = [c.name for c in qdrant.get_collections().collections]
    if COLLECTION_NAME not in existing_collections:
        qdrant.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=1024, distance=Distance.COSINE)
        )

    qdrant.upsert(collection_name=COLLECTION_NAME, points=points)
    print(f"✅ Uspešno indeksirano {len(points)} unosa zaposlenih u Qdrant kolekciju '{COLLECTION_NAME}'!")


if __name__ == "__main__":
    sync_employees_from_r2()