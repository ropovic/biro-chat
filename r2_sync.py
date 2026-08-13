import os
import json
import boto3
import streamlit as st
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance


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
        print("⚠️ Nedostaju Cloudflare R2 kredencijali u Secrets.")
        return None
        
    return boto3.client(
        service_name="s3",
        endpoint_url=f"https://{R2_ACCOUNT_ID}.r2.cloudflarestorage.com",
        aws_access_key_id=R2_ACCESS_KEY_ID,
        aws_secret_access_key=R2_SECRET_ACCESS_KEY,
        region_name="auto"
    )


def sync_employees_from_r2():
    s3 = get_s3_client()
    if not s3:
        print("⚠️ Preskačem R2 sinhronizaciju (nedostaju R2 ključevi).")
        return

    print("Preuzimanje 'employees.json' iz Cloudflare R2...")
    try:
        response = s3.get_object(Bucket=R2_BUCKET_NAME, Key="employees.json")
        raw_data = response['Body'].read().decode('utf-8')
        employees_data = json.loads(raw_data)
    except Exception as e:
        print(f"❌ Greška pri čitanju employees.json iz R2: {e}")
        return

    from sentence_transformers import SentenceTransformer
    embedder = SentenceTransformer("BAAI/bge-m3")
    qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, check_compatibility=False)

    # Provera i rekreiranje kolekcije ako dimenzija vektora nije 1024
    try:
        collection_info = qdrant.get_collection(collection_name=COLLECTION_NAME)
        current_size = collection_info.config.params.vectors.size
        if current_size != 1024:
            print(f"⚠️ Dimenzija kolekcije ({current_size}) se ne poklapa sa BGE-M3 (1024). Ponovo kreiram kolekciju...")
            qdrant.recreate_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(size=1024, distance=Distance.COSINE)
            )
    except Exception:
        qdrant.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=1024, distance=Distance.COSINE)
        )

    points = []
    for idx, emp in enumerate(employees_data):
        doc_type = emp.get("doc_type", "employee")
        photo_filename = emp.get("photo_filename", "")
        
        if R2_PUBLIC_URL_PREFIX:
            image_url = f"{R2_PUBLIC_URL_PREFIX}/{photo_filename}"
        else:
            image_url = photo_filename

        full_text = (
            f"Ime i prezime: {emp.get('name', '')}. "
            f"Rola / Funkcija / Pozicija: {emp.get('role', '')}. "
            f"Tip: {doc_type}. "
            f"Direktor: {'Da' if emp.get('is_director') else 'Ne'}. "
            f"Zamenik direktora: {'Da' if emp.get('is_deputy_director') else 'Ne'}. "
            f"Opis: {emp.get('description', '')}"
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
                id=20000 + idx,
                vector=vector,
                payload=payload
            )
        )

    qdrant.upsert(collection_name=COLLECTION_NAME, points=points)
    print(f"✅ Uspešno indeksirano {len(points)} unosa zaposlenih u Qdrant!")