import json
import boto3
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
from sentence_transformers import SentenceTransformer
import config

# Inicijalizacija R2 (S3) klijenta
s3 = boto3.client(
    service_name="s3",
    endpoint_url=f"https://{config.R2_ACCOUNT_ID}.r2.cloudflarestorage.com",
    aws_access_key_id=config.R2_ACCESS_KEY_ID,
    aws_secret_access_key=config.R2_SECRET_ACCESS_KEY,
    region_name="auto"
)

# Inicijalizacija Qdrant i Embedding modela
qdrant = QdrantClient(url=config.QDRANT_URL, api_key=config.QDRANT_API_KEY)
embedder = SentenceTransformer("all-MiniLM-L6-v2")

def sync_employees_from_r2():
    """
    Preuzima informacije o zaposlenima iz Cloudflare R2 bucket-a.
    Očekuje se da bucket sadrži 'employees.json' ili pojedinačne JSON opise
    uz odgovarajuće fotografije.
    """
    print("Preuzimanje podataka o zaposlenima iz Cloudflare R2...")
    
    try:
        # Primer 1: Preuzimanje centralnog manifesta 'employees.json'
        response = s3.get_object(Bucket=config.R2_BUCKET_NAME, Key="employees.json")
        employees_data = json.loads(response['Body'].read().decode('utf-8'))
    except Exception as e:
        print(f"Greška pri citanju employees.json: {e}")
        return

    points = []
    for idx, emp in enumerate(employees_data):
        # Formatiranje teksta za embedding
        full_text = (
            f"Ime i prezime: {emp['name']}. "
            f"Pozicija/Rola: {emp['role']}. "
            f"Da li je direktor: {'Da' if emp.get('is_director') else 'Ne'}. "
            f"Opis i biografija: {emp.get('description', '')}"
        )
        
        # Generisanje vektora
        vector = embedder.encode(full_text).tolist()
        
        # URL fotografije u Cloudflare Bucket-u
        image_url = f"{config.R2_PUBLIC_URL_PREFIX}/{emp['photo_filename']}"
        
        payload = {
            "doc_type": "employee_profile",
            "name": emp['name'],
            "role": emp['role'],
            "is_director": emp.get('is_director', False),
            "description": emp.get('description', ''),
            "image_url": image_url,
            "text": full_text
        }
        
        points.append(
            PointStruct(
                id=10000 + idx,  # Unikatni ID opseg za zaposlene
                vector=vector,
                payload=payload
            )
        )
    
    # Provera i kreiranje kolekcije ako ne postoji
    collections = [c.name for c in qdrant.get_collections().collections]
    if config.COLLECTION_NAME not in collections:
        qdrant.create_collection(
            collection_name=config.COLLECTION_NAME,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE)
        )
        
    # Upsert u Qdrant
    qdrant.upsert(collection_name=config.COLLECTION_NAME, points=points)
    print(f"Uspešno indeksirano {len(points)} zaposlenih u Qdrant bazu!")

if __name__ == "__main__":
    sync_employees_from_r2()