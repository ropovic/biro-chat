import os
import re
import unicodedata
import boto3
import streamlit as st
from rag_engine import ask_birochat

# ==========================================
# 1. KONFIGURACIJA CLOUDFLARE R2 BUCKETA
# ==========================================
R2_ACCOUNT_ID = os.getenv("R2_ACCOUNT_ID", "TVOJ_R2_ACCOUNT_ID")
R2_ACCESS_KEY_ID = os.getenv("R2_ACCESS_KEY_ID", "TVOJ_R2_ACCESS_KEY")
R2_SECRET_ACCESS_KEY = os.getenv("R2_SECRET_ACCESS_KEY", "TVOJ_R2_SECRET_KEY")
R2_BUCKET_NAME = os.getenv("R2_BUCKET_NAME", "fotografijebiro")
R2_PUBLIC_URL = os.getenv("R2_PUBLIC_URL", "https://pub-xxx.r2.dev")  # Tvoj javni R2 URL ili R2 custom domen

# Imena zamenika direktora prema tvojoj specifikaciji
DEPUTIES_NAMES = ["goran caldovic", "caldovic", "svetlana mihajlovic", "mihajlovic"]

def normalize_text(text: str) -> str:
    """Smanjuje slova i uklanja srpske dijakritike radi lakšeg poređenja imena."""
    text = text.lower().replace("đ", "dj")
    nfkd = unicodedata.normalize('NFKD', text)
    return "".join([c for c in nfkd if not unicodedata.combining(c)])

@st.cache_resource
def get_r2_client():
    """Konekcija na Cloudflare R2 preko boto3 S3 API-ja."""
    if not R2_ACCESS_KEY_ID or R2_ACCESS_KEY_ID == "TVOJ_R2_ACCESS_KEY":
        return None
    try:
        s3 = boto3.client(
            service_name="s3",
            endpoint_url=f"https://{R2_ACCOUNT_ID}.r2.cloudflarestorage.com",
            aws_access_key_id=R2_ACCESS_KEY_ID,
            aws_secret_access_key=R2_SECRET_ACCESS_KEY,
            region_name="auto"
        )
        return s3
    except Exception as e:
        st.error(f"Greška pri povezivanju na R2: {e}")
        return None

@st.cache_data(ttl=300)
def load_personnel_catalog_from_r2():
    """
    Skenira R2 bucket, spaja fotografije sa njihovim pratećim .txt opisima
    i pravi katalog osoblja.
    """
    s3 = get_r2_client()
    if not s3:
        return []

    try:
        response = s3.list_objects_v2(Bucket=R2_BUCKET_NAME)
        if "Contents" not in response:
            return []

        all_keys = [obj["Key"] for obj in response["Contents"]]
        
        image_files = [k for k in all_keys if k.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))]
        text_files = [k for k in all_keys if k.lower().endswith('.txt')]

        catalog = []

        for img_key in image_files:
            # Povezivanje slike i .txt fajla na osnovu reči u imenu
            img_norm = normalize_text(img_key)
            img_words = set(re.findall(r'\w+', img_norm)) - {'jpg', 'jpeg', 'png', 'webp'}

            matching_txt_key = None
            for txt_key in text_files:
                txt_norm = normalize_text(txt_key)
                txt_words = set(re.findall(r'\w+', txt_norm)) - {'txt', 'foto'}
                # Ako se ključne reči poklapaju (npr. brano i vamovic)
                if img_words and img_words.issubset(txt_words):
                    matching_txt_key = txt_key
                    break

            # Čitanje sadržaja .txt fajla ako postoji
            description_content = ""
            if matching_txt_key:
                try:
                    txt_obj = s3.get_object(Bucket=R2_BUCKET_NAME, Key=matching_txt_key)
                    description_content = txt_obj['Body'].read().decode('utf-8').strip()
                except Exception:
                    description_content = ""

            # Određivanje uloge
            combined_info_norm = normalize_text(f"{img_key} {matching_txt_key or ''} {description_content}")
            
            role = "zaposleni"
            if "direktor" in combined_info_norm and not any(dep in combined_info_norm for dep in DEPUTIES_NAMES):
                role = "direktor"
            elif any(dep in combined_info_norm for dep in DEPUTIES_NAMES) or "zamenik" in combined_info_norm:
                role = "zamenik"

            # Naslov/Opis za prikaz
            if description_content:
                caption = description_content
            elif matching_txt_key:
                # Izvlačenje lepog imena iz fajla
                caption = matching_txt_key.replace("Foto_", "").replace(".txt", "").replace("_", " ")
            else:
                caption = img_key.rsplit('.', 1)[0].replace("_", " ").title()

            catalog.append({
                "image_url": f"{R2_PUBLIC_URL.rstrip('/')}/{img_key}",
                "caption": caption,
                "role": role,
                "search_text": combined_info_norm
            })

        return catalog

    except Exception as e:
        st.error(f"Greška pri skeniranju R2 bucketa: {e}")
        return []

# ==========================================
# 2. PROVERA UPITA I FILTRIRANJE
# ==========================================
PERSONNEL_KEYWORDS = ["direktor", "direktora", "zamenik", "zamenika", "zamenici", "zaposlen", "zaposleni", "zaposlenih", "radnik", "radnici", "uprava", "slika", "fotografija"]

def is_personnel_query(question: str) -> bool:
    """Proverava da li se pitanje odnosi na osoblje."""
    q_norm = normalize_text(question)
    return any(re.search(r'\b' + re.escape(kw) + r'\b', q_norm) for kw in PERSONNEL_KEYWORDS)

def filter_personnel(catalog, query: str):
    """Filtrira katalog na osnovu postavljenog pitanja."""
    q_norm = normalize_text(query)

    if "direktor" in q_norm and not ("zamenik" in q_norm or "zamenika" in q_norm or "zamenici" in q_norm):
        return [p for p in catalog if p["role"] == "direktor"]

    if any(k in q_norm for k in ["zamenik", "zamenika", "zamenici", "goran", "caldovic", "svetlana", "mihajlovic"]):
        return [p for p in catalog if p["role"] == "zamenik" or any(dep in p["search_text"] for dep in DEPUTIES_NAMES)]

    # Ako traži konkretno ime ili sve zaposlene
    matched = [p for p in catalog if any(word in p["search_text"] for word in q_norm.split() if len(word) > 3)]
    return matched if matched else catalog

# ==========================================
# 3. STREAMLIT INTERFEJS
# ==========================================
st.set_page_config(page_title="BiroChat", page_icon="🌲", layout="wide")

st.title("🌲 BiroChat - Korporativni Asistent")
st.caption("Pretraga dokumentacije i uvid u zaposlene u Birou")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Prikaz istorije
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if "content" in msg and msg["content"]:
            st.markdown(msg["content"])
        
        if "images" in msg and msg["images"]:
            cols = st.columns(min(len(msg["images"]), 3))
            for idx, img in enumerate(msg["images"]):
                with cols[idx % 3]:
                    st.image(img["image_url"], caption=img["caption"], use_container_width=True)

# Unos korisnika
if user_input := st.chat_input("Postavite pitanje o dokumentima ili zaposlenima..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        # ROUTER: Ako je pitanje o osoblju -> Idi u R2, ignoriši tekstualne ugovore
        if is_personnel_query(user_input):
            st.info("🔍 Pretražujem registar fotografija u R2 bazi...")
            
            catalog = load_personnel_catalog_from_r2()
            filtered_photos = filter_personnel(catalog, user_input)
            
            if filtered_photos:
                answer_text = f"Pronađeno u registru fotografija Biroa:"
                st.markdown(answer_text)
                
                cols = st.columns(min(len(filtered_photos), 3))
                for idx, img in enumerate(filtered_photos):
                    with cols[idx % 3]:
                        st.image(img["image_url"], caption=img["caption"], use_container_width=True)
                
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer_text,
                    "images": filtered_photos
                })
            else:
                answer_text = "⚠️ U R2 bucketu nisu pronađene odgovarajuće fotografije."
                st.warning(answer_text)
                st.session_state.messages.append({"role": "assistant", "content": answer_text})
        
        else:
            # KLASIČAN RAG ZA TEKSTUALNU DOKUMENTACIJU
            with st.spinner("Pretražujem tekstualnu dokumentaciju..."):
                result = ask_birochat(user_input)
                answer_text = result.get("answer", "Nema odgovora.")
                sources = result.get("sources", [])
                
                st.markdown(answer_text)
                
                if sources:
                    with st.expander("📚 Korišćeni izvori iz baze"):
                        for src in sources:
                            st.write(f"- `{src}`")
                            
                st.session_state.messages.append({"role": "assistant", "content": answer_text})