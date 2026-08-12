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
    """
    Preuzima 'employees.json' iz Cloudflare R2 bucket-a i indeksira
    sve zaposlene u Qdrant. Preskače ako ključevi nisu podešeni.
    """
    s3 = get_s3_client()
    if not s3:
        print("⚠️ Preskačem R2 sinhronizaciju jer R2 ključevi nisu definisani u Secrets.")
        return

    print("Preuzimanje manifesta 'employees.json' iz Cloudflare R2...")
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
        
        if R2_PUBLIC_URL_PREFIX:
            image_url = f"{R2_PUBLIC_URL_PREFIX}/{photo_filename}"
        else:
            image_url = photo_filename

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
                id=20000 + idx,
                vector=vector,
                payload=payload
            )
        )

    existing_collections = [c.name for c in qdrant.get_collections().collections]
    if COLLECTION_NAME not in existing_collections:
        qdrant.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=1024, distance=Distance.COSINE)
        )

    qdrant.upsert(collection_name=COLLECTION_NAME, points=points)
    print(f"✅ Uspešno indeksirano {len(points)} unosa u Qdrant!")