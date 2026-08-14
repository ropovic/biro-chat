import os
import boto3
from botocore.config import Config
import streamlit as st
from qdrant_client import QdrantClient
from llama_index.core import VectorStoreIndex, Settings
from llama_index.vector_stores.qdrant import QdrantVectorStore
# Prilagodi importove za tvoj LLM i embedding model (ukoliko koristiš multilingual embeddings kao ranije, zameni ovde)
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI

st.set_page_config(page_title="BiroChat", page_icon="🤖", layout="wide")

# ==========================================
# 1. BRZI UPITI (6 DUGMIĆA)
# ==========================================
QUICK_PROMPTS = [
    "Ko je direktor?",
    "Ko su zamenici direktora?",
    "Spisak zaposlenih",
    "Štampači u Birou",
    "Toneri za štampače",
    "Osnove gazdovanja u Birou"
]

# ==========================================
# 2. INICIJALIZACIJA BAZE (QDRANT)
# ==========================================
@st.cache_resource
def init_query_engine():
    """Inicijalizacija LlamaIndex + Qdrant vektorske baze."""
    Settings.llm = OpenAI(model="gpt-4o", api_key=os.environ.get("OPENAI_API_KEY"))
    Settings.embed_model = OpenAIEmbedding(api_key=os.environ.get("OPENAI_API_KEY"))
    
    client = QdrantClient(
        url=os.environ.get("QDRANT_URL", "tvoj-qdrant-url"),
        api_key=os.environ.get("QDRANT_API_KEY", "tvoj-kljuc")
    )
    vector_store = QdrantVectorStore(client=client, collection_name="biro_baza")
    index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
    return index.as_query_engine(similarity_top_k=3)

# ==========================================
# 3. IMPLEMENTACIJA CLOUDFLARE R2 (BOTO3)
# ==========================================
def fetch_photos_from_r2(query_text: str):
    """
    Povezuje se na Cloudflare R2 putem Boto3 biblioteke i povlači fotografije
    na osnovu traženog upita (direktor, zamenici, zaposleni).
    """
    # Preuzimanje Cloudflare akreditiva iz sistemskih varijabli (ili Streamlit secrets)
    account_id = os.environ.get("CF_ACCOUNT_ID")
    access_key = os.environ.get("CF_ACCESS_KEY")
    secret_key = os.environ.get("CF_SECRET_KEY")
    bucket_name = os.environ.get("CF_BUCKET_NAME", "biro-mediji")  # Zameni pravim imenom bucketa
    
    if not all([account_id, access_key, secret_key]):
        st.warning("⚠️ R2 Akreditivi nedostaju u okruženju (CF_ACCOUNT_ID, CF_ACCESS_KEY, CF_SECRET_KEY).")
        return []

    # Kreiranje klijenta za Cloudflare R2
    s3_client = boto3.client(
        's3',
        endpoint_url=f"https://{account_id}.r2.cloudflarestorage.com",
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        config=Config(signature_version='s3v4'),
        region_name='auto' # Za Cloudflare R2 obavezno auto
    )

    image_urls = []
    
    try:
        # Određivanje prefiksa na osnovu ključnih reči u upitu.
        # Možeš prilagoditi foldere tačno onako kako se zovu u R2 bucketu.
        q = query_text.lower()
        if "zamenic" in q:
            prefix = "zamenici/"
        elif "direktor" in q:
            prefix = "direktor/"
        else:
            prefix = "zaposleni/"

        # Preuzimanje liste fajlova iz foldera
        response = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
        
        if 'Contents' in response:
            for obj in response['Contents']:
                file_key = obj['Key']
                # Filtriranje samo slikovnih formata
                if file_key.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                    # Generisanje privremenog (presigned) linka koji važi 1 sat
                    url = s3_client.generate_presigned_url(
                        ClientMethod='get_object',
                        Params={'Bucket': bucket_name, 'Key': file_key},
                        ExpiresIn=3600
                    )
                    image_urls.append(url)
                    
        return image_urls
    except Exception as e:
        st.error(f"⚠️ Došlo je do greške prilikom povezivanja sa R2: {str(e)}")
        return []

def requires_r2_photos(query_text: str) -> bool:
    """Određuje da li upit zahteva aktivaciju Boto3 pretrage za slikama."""
    keywords = ["direktor", "zamenic", "zaposlen"]
    return any(kw in query_text.lower() for kw in keywords)


# ==========================================
# 4. INICIJALIZACIJA SESSION STATE-A
# ==========================================
if "query_engine" not in st.session_state:
    try:
        st.session_state.query_engine = init_query_engine()
    except Exception as e:
        st.session_state.query_engine = None
        st.error(f"Greška pri povezivanju sa vektorskom bazom: {e}")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending_prompt" not in st.session_state:
    st.session_state.pending_prompt = None


# ==========================================
# 5. RENDEROVANJE INTERFEJSA
# ==========================================
st.title("🤖 BiroChat")
st.caption("Interni pretraživač Biroa (Qdrant Tekst + Cloudflare R2 Mediji)")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if "images" in msg and msg["images"]:
            # Dinamički kreira kolone ako ima više slika zaposlenih
            cols = st.columns(min(len(msg["images"]), 3)) 
            for img_idx, img_url in enumerate(msg["images"]):
                cols[img_idx % 3].image(img_url, use_container_width=True)
        st.markdown(msg["content"])

st.write("---")
st.caption("Brzi upiti (klikni za automatsku pretragu):")
row1 = st.columns(3)
row2 = st.columns(3)

for i, prompt_text in enumerate(QUICK_PROMPTS):
    col = row1[i] if i < 3 else row2[i - 3]
    if col.button(prompt_text, use_container_width=True):
        st.session_state.pending_prompt = prompt_text
        st.rerun()

# ==========================================
# 6. OBRADA KORISNIČKOG UPITA (CHAT INPUT)
# ==========================================
user_input = st.chat_input("Postavi pitanje...")

if st.session_state.pending_prompt:
    active_query = st.session_state.pending_prompt
    st.session_state.pending_prompt = None
elif user_input:
    active_query = user_input
else:
    active_query = None

if active_query:
    st.session_state.messages.append({"role": "user", "content": active_query})
    with st.chat_message("user"):
        st.markdown(active_query)

    with st.chat_message("assistant"):
        found_images = []
        
        # --- 1. R2 BUCKET PRETRAGA ---
        if requires_r2_photos(active_query):
            st.markdown("🔍 *Pretražujem registar fotografija u R2 bazi...*")
            found_images = fetch_photos_from_r2(active_query)
            
            if found_images:
                st.success(f"Pronađeno {len(found_images)} fotografija.")
                cols = st.columns(min(len(found_images), 3))
                for img_idx, img_url in enumerate(found_images):
                    cols[img_idx % 3].image(img_url, use_container_width=True)
            else:
                st.warning("⚠️ U R2 bucketu nisu pronađene odgovarajuće fotografije za ovaj upit.")
        
        # --- 2. QDRANT TEKSTUALNA PRETRAGA ---
        with st.spinner("Pretražujem Biro dokumentaciju (Qdrant)..."):
            if st.session_state.query_engine:
                try:
                    response = st.session_state.query_engine.query(active_query)
                    text_answer = str(response)
                except Exception as e:
                    text_answer = f"Greška pri dobavljanju teksta iz baze: {str(e)}"
            else:
                text_answer = "⚠️ Nema konekcije sa vektorskom bazom (Qdrant)."
            
            st.markdown(text_answer)
            
    # Čuvanje rezultata u memoriju chata
    st.session_state.messages.append({
        "role": "assistant", 
        "content": text_answer,
        "images": found_images
    })