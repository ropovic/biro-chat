import os
from dotenv import load_dotenv

load_dotenv()

# Cloudflare R2 Podešavanja
R2_ACCOUNT_ID = os.getenv("R2_ACCOUNT_ID")
R2_ACCESS_KEY_ID = os.getenv("R2_ACCESS_KEY_ID")
R2_SECRET_ACCESS_KEY = os.getenv("R2_SECRET_ACCESS_KEY")
R2_BUCKET_NAME = os.getenv("R2_BUCKET_NAME")
R2_PUBLIC_URL_PREFIX = os.getenv("R2_PUBLIC_URL_PREFIX")  # npr. https://pub-xxx.r2.dev ili custom domen

# Qdrant & Groq
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
COLLECTION_NAME = "biro_documents"