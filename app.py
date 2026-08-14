import os
import re
import unicodedata
import boto3
import streamlit as st
from rag_engine import ask_birochat

st.set_page_config(page_title="BiroChat", page_icon="🌲", layout="wide")

st.markdown("""
<style>
.stApp { background-color: #f2f7f2; }
[data-testid="stSidebar"] { background-color: #e5f0e5; border-right: 1px solid #d4e5d4; }
.title-text { color: #2e7d32; font-weight: 800; font-family: 'Segoe UI', sans-serif; margin-bottom: 0px; }
.stButton button { background-color: #2e7d32; color: white; border: none; }
.stButton button:hover { background-color: #1b5e20; color: white; }
</style>
""", unsafe_allow_html=True)

R2_ACCOUNT_ID = os.getenv("R2_ACCOUNT_ID", "TVOJ_R2_ACCOUNT_ID")
R2_ACCESS_KEY_ID = os.getenv("R2_ACCESS_KEY_ID", "TVOJ_R2_ACCESS_KEY")
R2_SECRET_ACCESS_KEY = os.getenv("R2_SECRET_ACCESS_KEY", "TVOJ_R2_SECRET_KEY")
R2_BUCKET_NAME = os.getenv("R2_BUCKET_NAME", "fotografijebiro")

DEPUTIES_TOKENS = ["caldovic", "mihajlovic", "goran", "svetlana"]

def cyrillic_to_latin(text: str) -> str:
    cyr_map = {
        'а':'a', 'б':'b', 'в':'v', 'г':'g', 'д':'d', 'ђ':'dj', 'е':'e', 'ж':'z',
        'з':'z', 'и':'i', 'ј':'j', 'к':'k', 'л':'l', 'љ':'lj', 'м':'m', 'н':'n',
        'њ':'nj', 'о':'o', 'п':'p', 'р':'r', 'с':'s', 'т':'t', 'ћ':'c', 'у':'u',
        'ф':'f', 'х':'h', 'ц':'c', 'ч':'c', 'џ':'dz', 'ш':'s',
        'А':'A', 'Б':'B', 'В':'V', 'Г':'G', 'Д':'D', 'Ђ':'Dj', 'Е':'E', 'Ж':'Z',
        'З':'Z', 'И':'I', 'Ј':'J', 'К':'K', 'Л':'L', 'Љ':'Lj', 'М':'M', 'Н':'N',
        'Њ':'Nj', 'О':'O', 'П':'P', 'Р':'R', 'С':'S', 'Т':'T', 'Ћ':'C', 'У':'U',
        'Ф':'F', 'Х':'H', 'Ц':'C', 'Ч':'C', 'Џ':'Dz', 'Ш':'S'
    }
    for cyr, lat in cyr_map.items():
        text = text.replace(cyr, lat)
    return text

def normalize_text(text: str) -> str:
    if not text: return ""
    text = cyrillic_to_latin(text).lower().replace("đ", "dj")
    nfkd = unicodedata.normalize('NFKD', text)
    return "".join([c for c in nfkd if not unicodedata.combining(c)])

@st.cache_resource
def get_r2_client():
    if not R2_ACCESS_KEY_ID or R2_ACCESS_KEY_ID == "TVOJ_R2_ACCESS_KEY": return None
    try:
        return boto3.client(
            service_name="s3", endpoint_url=f"https://{R2_ACCOUNT_ID}.r2.cloudflarestorage.com",
            aws_access_key_id=R2_ACCESS_KEY_ID, aws_secret_access_key=R2_SECRET_ACCESS_KEY, region_name="auto"
        )
    except Exception as e:
        st.error(f"Greška pri povezivanju na Cloudflare R2: {e}")
        return None

def get_presigned_url(s3_client, key: str) -> str:
    try:
        return s3_client.generate_presigned_url('get_object', Params={'Bucket': R2_BUCKET_NAME, 'Key': key}, ExpiresIn=3600)
    except Exception:
        return ""

@st.cache_data(ttl=3600)
def fetch_system_logos():
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
def load_media_catalog_from_r2():
    s3 = get_r2_client()
    if not s3: return []

    try:
        response = s3.list_objects_v2(Bucket=R2_BUCKET_NAME)
        if "Contents" not in response: return []

        all_keys = [obj["Key"] for obj in response["Contents"]]
        image_files = [k for k in all_keys if k.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')) and 'logo' not in k.lower()]
        text_files = [k for k in all_keys if k.lower().endswith('.txt')]

        catalog = []
        seen_identities = set()

        for img_key in image_files:
            img_base = os.path.splitext(img_key)[0]
            img_norm = normalize_text(img_base).replace("_", " ").replace("-", " ")

            matching_txt_key = None
            for txt_key in text_files:
                txt_base = os.path.splitext(txt_key)[0]
                if normalize_text(txt_base) in img_norm or img_norm in normalize_text(txt_base):
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

            # Detekcija kategorije: Dijagram / Ruža vetrova VS Osoblje
            is_diagram = any(k in search_corpus for k in ["ruza", "vetro", "vetar", "dijagram", "grafik", "karta", "skica"])
            
            is_deputy = any(dep in search_corpus for dep in DEPUTIES_TOKENS) or "zamenik" in search_corpus
            is_director = "direktor" in search_corpus and not is_deputy
            is_personnel = (is_director or is_deputy or "zaposlen" in search_corpus or "radnik" in search_corpus or "foto" in search_corpus) and not is_diagram

            category = "diagram" if is_diagram else ("personnel" if is_personnel else "other")
            role = "direktor" if is_director else ("zamenik" if is_deputy else "zaposleni")

            if category == "personnel":
                if role == "direktor":
                    person_id = "direktor_glavni"
                elif "caldovic" in search_corpus or "goran" in search_corpus:
                    person_id = "zamenik_caldovic"
                elif "mihajlovic" in search_corpus or "svetlana" in search_corpus:
                    person_id = "zamenik_mihajlovic"
                else:
                    person_id = img_norm.strip()

                if person_id in seen_identities:
                    continue
                seen_identities.add(person_id)
            else:
                person_id = img_key

            title = img_base.replace("_", " ").replace("Foto", "").replace("foto", "").strip().title()
            if not title or len(title) < 3:
                title = person_id.replace("_", " ").title()

            if role == "direktor" and "direktor" not in title.lower():
                title += " (Direktor)"
            elif role == "zamenik" and "zamenik" not in title.lower():
                title += " (Zamenik direktora)"

            image_presigned_url = get_presigned_url(s3, img_key)

            catalog.append({
                "image_key": img_key,
                "image_url": image_presigned_url,
                "title": title,
                "description": description_content,
                "category": category,
                "role": role,
                "search_corpus": search_corpus
            })

        return catalog
    except Exception as e:
        st.error(f"Greška pri skeniranju R2 bucketa: {e}")
        return []

# ==========================================
# 3. ZAGLAVLJE I SIDEBAR
# ==========================================
system_logos = fetch_system_logos()

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

with st.sidebar:
    st.markdown("### ⚙️ Opcije")
    if st.button("🧹 Obriši poruke", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

MEDIA_KEYWORDS = [
    "direktor", "direktora", "zamenik", "zamenika", "zamenici", 
    "zaposlen", "zaposleni", "zaposlenih", "radnik", "radnici", 
    "uprava", "slika", "slike", "fotografija", "fotografije", 
    "ruza", "ruze", "vetar", "vetrova", "dijagram", "dijagrami", 
    "grafik", "grafikoni", "karta", "karte", "prikaz"
]

def is_media_query(question: str) -> bool:
    q_norm = normalize_text(question)
    return any(re.search(r'\b' + re.escape(kw) + r'\b', q_norm) for kw in MEDIA_KEYWORDS)

def filter_media(catalog, query: str):
    q_norm = normalize_text(query)

    # 1. Traženje dijagrama ruža vetrova
    if any(k in q_norm for k in ["ruza", "ruze", "vetar", "vetrova", "dijagram", "dijagrami"]):
        diagrams = [p for p in catalog if p["category"] == "diagram" or any(w in p["search_corpus"] for w in ["ruza", "vetro", "vetar", "dijagram"])]
        return diagrams

    # 2. Traženje direktora
    if "direktor" in q_norm and not any(k in q_norm for k in ["zamenik", "zamenika", "zamenici"]):
        res = [p for p in catalog if p["role"] == "direktor"]
        if res: return res

    # 3. Traženje zamenika
    if any(k in q_norm for k in ["zamenik", "zamenika", "zamenici", "goran", "caldovic", "svetlana", "mihajlovic"]):
        res = [p for p in catalog if p["role"] == "zamenik"]
        if res: return res

    # 4. Traženje svih zaposlenih (VRAĆA SAMO OSOBLJE, NIKADA DIJAGRAME!)
    if any(k in q_norm for k in ["zaposlen", "zaposleni", "radnik", "radnici", "osoblje"]):
        personnel = [p for p in catalog if p["category"] == "personnel" or p["role"] in ["direktor", "zamenik", "zaposleni"]]
        return personnel

    # 5. Opšta pretraga po rečima u korpusu
    query_words = [w for w in q_norm.split() if len(w) > 2 and w not in ["ko", "je", "su", "u", "biro", "biroa", "pokazi", "daj", "slike"]]
    matched = [p for p in catalog if any(word in p["search_corpus"] for word in query_words)]
    return matched

# ==========================================
# 4. CHAT INTERFEJS
# ==========================================
if "messages" not in st.session_state:
    st.session_state.messages = []

st.markdown("##### 💡 Brza pitanja:")
quick_questions = [
    "Ko je direktor Biroa?",
    "Ko su zaposleni u Birou?",
    "Pokaži dijagrame ruža vetrova.",
    "Koji štampači se koriste u Birou?",
    "Navedi član 14 Kolektivnog ugovora.",
    "Spisak opreme i tonera"
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
                st.image(img["image_url"], caption=img["title"], width=300)

chat_input_val = st.chat_input("Postavite pitanje o dokumentima ili zaposlenima...")
user_input = selected_quick_q or chat_input_val

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        if is_media_query(user_input):
            st.info("🔍 Pretražujem registar medija i fotografija u R2 bazi...")
            catalog = load_media_catalog_from_r2()
            
            if not catalog:
                answer_text = "⚠️ Nije moguće pristupiti R2 bucketu ili bucket nema sadržaja."
                st.warning(answer_text)
                st.session_state.messages.append({"role": "assistant", "content": answer_text})
            else:
                filtered_media = filter_media(catalog, user_input)
                if filtered_media:
                    answer_text = f"Pronađeno u registru R2 baze ({len(filtered_media)} stavki):"
                    st.markdown(answer_text)
                    for img in filtered_media:
                        st.image(img["image_url"], caption=img["title"], width=300)
                    
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer_text,
                        "images": filtered_media
                    })
                else:
                    answer_text = "⚠️ U R2 bucketu nisu pronađene odgovarajuće fotografije ili dijagrami."
                    st.warning(answer_text)
                    st.session_state.messages.append({"role": "assistant", "content": answer_text})
        
        else:
            with st.spinner("Pretražujem bazu dokumenata..."):
                result = ask_birochat(user_input)
                answer_text = result.get("answer", "Nema odgovora.")
                sources = result.get("sources", [])
                
                st.markdown(answer_text)
                
                if sources:
                    with st.expander("📚 Korišćeni izvori iz baze / weba"):
                        for src in sources:
                            st.write(f"- `{src}`")
                            
                st.session_state.messages.append({"role": "assistant", "content": answer_text})