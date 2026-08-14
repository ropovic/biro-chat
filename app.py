import os
import re
import unicodedata
import boto3
import streamlit as st
from rag_engine import ask_birochat

# ==========================================
# 1. PODEŠAVANJA STRANICE I DIZAJNA
# ==========================================
st.set_page_config(page_title="BiroChat", page_icon="🌲", layout="wide")

st.markdown("""
<style>
/* Svetla zelena pozadina za aplikaciju u tonu Srbijašuma */
.stApp {
    background-color: #f2f7f2;
}
/* Bočna traka sa blagim kontrastom */
[data-testid="stSidebar"] {
    background-color: #e5f0e5;
    border-right: 1px solid #d4e5d4;
}
/* Stilizacija naslova */
.title-text {
    color: #2e7d32;
    font-weight: 800;
    font-family: 'Segoe UI', sans-serif;
    margin-bottom: 0px;
}
/* Stilizacija dugmadi */
.stButton button {
    background-color: #2e7d32;
    color: white;
    border: none;
}
.stButton button:hover {
    background-color: #1b5e20;
    color: white;
}
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. KONFIGURACIJA CLOUDFLARE R2 BUCKETA
# ==========================================
R2_ACCOUNT_ID = os.getenv("R2_ACCOUNT_ID", "TVOJ_R2_ACCOUNT_ID")
R2_ACCESS_KEY_ID = os.getenv("R2_ACCESS_KEY_ID", "TVOJ_R2_ACCESS_KEY")
R2_SECRET_ACCESS_KEY = os.getenv("R2_SECRET_ACCESS_KEY", "TVOJ_R2_SECRET_KEY")
R2_BUCKET_NAME = os.getenv("R2_BUCKET_NAME", "fotografijebiro")

DEPUTIES_TOKENS = ["caldovic", "mihajlovic", "goran caldovic", "svetlana mihajlovic"]

def normalize_text(text: str) -> str:
    if not text: return ""
    text = text.lower().replace("đ", "dj")
    nfkd = unicodedata.normalize('NFKD', text)
    return "".join([c for c in nfkd if not unicodedata.combining(c)])

@st.cache_resource
def get_r2_client():
    if not R2_ACCESS_KEY_ID or R2_ACCESS_KEY_ID == "TVOJ_R2_ACCESS_KEY": return None
    try:
        return boto3.client(
            service_name="s3",
            endpoint_url=f"https://{R2_ACCOUNT_ID}.r2.cloudflarestorage.com",
            aws_access_key_id=R2_ACCESS_KEY_ID,
            aws_secret_access_key=R2_SECRET_ACCESS_KEY,
            region_name="auto"
        )
    except Exception as e:
        st.error(f"Greška pri povezivanju na Cloudflare R2: {e}")
        return None

def get_presigned_url(s3_client, key: str) -> str:
    try:
        return s3_client.generate_presigned_url(
            'get_object', Params={'Bucket': R2_BUCKET_NAME, 'Key': key}, ExpiresIn=3600
        )
    except Exception:
        return ""

@st.cache_data(ttl=3600)
def fetch_system_logos():
    """Traži Biro_logo i Srbijasume_logo i vraća njihove URL-ove."""
    s3 = get_r2_client()
    logos = {"biro": None, "srbijasume": None}
    if not s3: return logos
    
    try:
        response = s3.list_objects_v2(Bucket=R2_BUCKET_NAME)
        if "Contents" in response:
            for obj in response["Contents"]:
                key_lower = obj["Key"].lower()
                if "biro_logo" in key_lower:
                    logos["biro"] = get_presigned_url(s3, obj["Key"])
                elif "srbijasume_logo" in key_lower:
                    logos["srbijasume"] = get_presigned_url(s3, obj["Key"])
    except Exception:
        pass
    return logos

@st.cache_data(ttl=300)
def load_personnel_catalog_from_r2():
    s3 = get_r2_client()
    if not s3: return []

    try:
        response = s3.list_objects_v2(Bucket=R2_BUCKET_NAME)
        if "Contents" not in response: return []

        all_keys = [obj["Key"] for obj in response["Contents"]]
        
        # Izdvajamo zaposlene (ISKLJUČUJEMO fajlove koji u nazivu imaju 'logo')
        image_files = [k for k in all_keys if k.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')) and 'logo' not in k.lower()]
        text_files = [k for k in all_keys if k.lower().endswith('.txt')]

        catalog = []
        for img_key in image_files:
            img_base = os.path.splitext(img_key)[0]
            img_norm = normalize_text(img_base).replace("_", " ").replace("-", " ")
            img_tokens = [w for w in img_norm.split() if len(w) > 1]

            matching_txt_key = None
            for txt_key in text_files:
                txt_norm = normalize_text(txt_key).replace("_", " ").replace("-", " ")
                if all(token in txt_norm for token in img_tokens):
                    matching_txt_key = txt_key
                    break

            description_content = ""
            if matching_txt_key:
                try:
                    txt_obj = s3.get_object(Bucket=R2_BUCKET_NAME, Key=matching_txt_key)
                    description_content = txt_obj['Body'].read().decode('utf-8', errors='ignore').strip()
                except Exception:
                    pass

            search_corpus = normalize_text(f"{img_key} {matching_txt_key or ''} {description_content}")

            is_deputy = any(dep in search_corpus for dep in DEPUTIES_TOKENS) or "zamenik" in search_corpus
            is_director = "direktor" in search_corpus and not is_deputy

            role = "direktor" if is_director else ("zamenik" if is_deputy else "zaposleni")

            if description_content:
                title = description_content.split("–")[0].split("-")[0].strip()
                if len(title) > 40: title = img_base.replace("_", " ").title()
            else:
                title = img_base.replace("_", " ").title()

            if role == "direktor" and "direktor" not in title.lower(): title += " (Direktor)"
            elif role == "zamenik" and "zamenik" not in title.lower(): title += " (Zamenik direktora)"

            image_presigned_url = get_presigned_url(s3, img_key)

            catalog.append({
                "image_key": img_key,
                "image_url": image_presigned_url,
                "title": title,
                "description": description_content,
                "role": role,
                "search_corpus": search_corpus
            })

        return catalog
    except Exception as e:
        st.error(f"Greška pri skeniranju R2 bucketa '{R2_BUCKET_NAME}': {e}")
        return []

# ==========================================
# 3. ZAGLAVLJE I SIDEBAR
# ==========================================
system_logos = fetch_system_logos()

# Zaglavlje sa logotipima
col_l1, col_l2, col_l3 = st.columns([1, 4, 1])
with col_l1:
    if system_logos["biro"]:
        st.image(system_logos["biro"], use_container_width=True)
with col_l2:
    st.markdown("<h1 class='title-text' style='text-align: center;'>🌲 BiroChat</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #2e7d32; font-size: 1.2rem; font-weight: bold;'>Korporativni Asistent</p>", unsafe_allow_html=True)
with col_l3:
    if system_logos["srbijasume"]:
        st.image(system_logos["srbijasume"], use_container_width=True)

st.markdown("---")

# Bočna traka (Sidebar)
with st.sidebar:
    st.markdown("### ⚙️ Opcije")
    if st.button("🧹 Obriši poruke", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
        
    st.markdown("---")
    st.markdown("### 🧠 Sistemski moduli")
    st.info("""
    **LLM Model:**
    Groq Llama-3.1-8b-instant
    
    **Vektorska Baza:**
    Qdrant Vector Store
    
    **Embedder:**
    paraphrase-multilingual-MiniLM-L12-v2
    
    **Orkestrator:**
    LangChain
    
    **Web Ekstenzija:**
    Tavily API
    """)

# ==========================================
# 4. CHAT I LOGIKA PRETRAGE
# ==========================================
if "messages" not in st.session_state:
    st.session_state.messages = []

PERSONNEL_KEYWORDS = ["direktor", "direktora", "zamenik", "zamenika", "zamenici", "zaposlen", "zaposleni", "zaposlenih", "radnik", "radnici", "uprava", "slika", "fotografija"]

def is_personnel_query(question: str) -> bool:
    q_norm = normalize_text(question)
    return any(re.search(r'\b' + re.escape(kw) + r'\b', q_norm) for kw in PERSONNEL_KEYWORDS)

def filter_personnel(catalog, query: str):
    q_norm = normalize_text(query)
    if "direktor" in q_norm and not any(k in q_norm for k in ["zamenik", "zamenika", "zamenici"]):
        res = [p for p in catalog if p["role"] == "direktor"]
        if res: return res
    if any(k in q_norm for k in ["zamenik", "zamenika", "zamenici", "goran", "caldovic", "svetlana", "mihajlovic"]):
        res = [p for p in catalog if p["role"] == "zamenik"]
        if res: return res

    query_words = [w for w in q_norm.split() if len(w) > 2 and w not in ["ko", "je", "su", "u", "biro", "biroa"]]
    matched = [p for p in catalog if any(word in p["search_corpus"] for word in query_words)]
    return matched if matched else catalog

st.markdown("##### 💡 Brza pitanja:")
quick_questions = [
    "Ko je direktor Biroa?",
    "Ko su zamenici direktora?",
    "Koji štampači se koriste u Birou?",
    "Kolektivni ugovor - godišnji odmor",
    "Spisak opreme i tonera",
    "Ko je ministar zdravstva u Srbiji?"
]

col1, col2, col3 = st.columns(3)
q_cols = [col1, col2, col3, col1, col2, col3]

selected_quick_q = None
for idx, q in enumerate(quick_questions):
    if q_cols[idx].button(q, key=f"quick_btn_{idx}", use_container_width=True):
        selected_quick_q = q

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if "content" in msg and msg["content"]:
            st.markdown(msg["content"])
        if "images" in msg and msg["images"]:
            for img in msg["images"]:
                c1, c2 = st.columns([1, 2])
                with c1:
                    if img["image_url"]:
                        st.image(img["image_url"], caption=img["title"], use_container_width=True)
                    else:
                        st.warning("⚠️ Slika nije dostupna.")
                with c2:
                    st.subheader(img["title"])
                    if img["description"]:
                        st.write(img["description"])

chat_input_val = st.chat_input("Postavite pitanje o dokumentima ili zaposlenima...")
user_input = selected_quick_q or chat_input_val

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        if is_personnel_query(user_input):
            st.info("🔍 Pretražujem registar fotografija u R2 bazi...")
            catalog = load_personnel_catalog_from_r2()
            
            if not catalog:
                answer_text = "⚠️ Nije moguće pristupiti R2 bucketu ili bucket nema slika."
                st.warning(answer_text)
                st.session_state.messages.append({"role": "assistant", "content": answer_text})
            else:
                filtered_photos = filter_personnel(catalog, user_input)
                if filtered_photos:
                    answer_text = f"Pronađeno u registru fotografija Biroa:"
                    st.markdown(answer_text)
                    for img in filtered_photos:
                        c1, c2 = st.columns([1, 2])
                        with c1:
                            if img["image_url"]:
                                st.image(img["image_url"], caption=img["title"], use_container_width=True)
                            else:
                                st.warning("⚠️ Slika nije dostupna.")
                        with c2:
                            st.subheader(img["title"])
                            if img["description"]:
                                st.write(img["description"])
                    
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
            with st.spinner("Pretražujem bazu i web..."):
                result = ask_birochat(user_input)
                answer_text = result.get("answer", "Nema odgovora.")
                sources = result.get("sources", [])
                
                st.markdown(answer_text)
                
                if sources:
                    with st.expander("📚 Korišćeni izvori iz baze / weba"):
                        for src in sources:
                            st.write(f"- `{src}`")
                            
                st.session_state.messages.append({"role": "assistant", "content": answer_text})