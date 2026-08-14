import os
import re
import unicodedata
import streamlit as st
from rag_engine import ask_birochat

# ==========================================
# 1. PODEŠAVANJA STRANICE I DIZAJNA
# ==========================================
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

# ==========================================
# 2. HARDKODOVANI LOGOTIPI I KATALOG ZAPOSLENIH
# ==========================================
SYSTEM_LOGOS = {
    "biro": "https://pub-49fb3cc788a74e0a9edbac7e11305b94.r2.dev/Biro%20Logo.jpg",
    "srbijasume": "https://pub-49fb3cc788a74e0a9edbac7e11305b94.r2.dev/Srbija%C5%A1ume%20logo.jpg"
}

PERSONNEL_CATALOG = [
    {
        "title": "Brane Vamović, direktor",
        "full_title": "Fotografija Brane Vamovića, direktor u Birou za planiranje, PD Srbijašume",
        "image_url": "https://pub-49fb3cc788a74e0a9edbac7e11305b94.r2.dev/Brano%20Vamovi%C4%87%20-%20direktor.jpg",
        "role": "direktor",
        "search_corpus": "fotografija brane vamovića, direktor u birou za planiranje, pd srbijašume brano vamović direktor"
    },
    {
        "title": "Goran Ćaldović, viši projektant",
        "full_title": "Fotografija Gorana Ćaldovića, viši projektant u Birou za planiranje, PD Srbijašume",
        "image_url": "https://pub-49fb3cc788a74e0a9edbac7e11305b94.r2.dev/Goran%20%C4%86aldovi%C4%87.jpg",
        "role": "zamenik",
        "search_corpus": "fotografija gorana ćaldovića, viši projektant u birou za planiranje, pd srbijašume goran ćaldović zamenik"
    },
    {
        "title": "Svetlana Mihajlović, samostalni projektant",
        "full_title": "Fotografija Svetlane Mihajlović, samostalni projektant u Birou za planiranje, PD Srbijašume",
        "image_url": "https://pub-49fb3cc788a74e0a9edbac7e11305b94.r2.dev/Svetlana%20Mihajlovi%C4%87.jpg",
        "role": "zamenik",
        "search_corpus": "fotografija svetlane mihajlović, samostalni projektant u birou za planiranje, pd srbijašume svetlana mihajlović zamenik"
    },
    {
        "title": "Aleksandra Katić",
        "full_title": "Fotografija Aleksandre Katić, samostalni projektant u Birou za planiranje, PD Srbijašume",
        "image_url": "https://pub-49fb3cc788a74e0a9edbac7e11305b94.r2.dev/Aleksandra%20Kati%C4%87.jpg",
        "role": "zaposleni",
        "search_corpus": "fotografija aleksandre katić, samostalni projektant u birou za planiranje, pd srbijašume"
    },
    {
        "title": "Arsenije Simić",
        "full_title": "Fotografija Arsenija Simića, samostalni projektant u Birou za planiranje, PD Srbijašume",
        "image_url": "https://pub-49fb3cc788a74e0a9edbac7e11305b94.r2.dev/Arsenije%20Simi%C4%87.jpg",
        "role": "zaposleni",
        "search_corpus": "fotografija arsenija simića, samostalni projektant u birou za planiranje, pd srbijašume"
    },
    {
        "title": "Biljana Mirković",
        "full_title": "Fotografija Biljane Mirković, šef računovodstva u Birou za planiranje, PD Srbijašume",
        "image_url": "https://pub-49fb3cc788a74e0a9edbac7e11305b94.r2.dev/Biljana%20Mirkovi%C4%87.jpg",
        "role": "zaposleni",
        "search_corpus": "fotografija biljane mirković, šef računovodstva u birou za planiranje, pd srbijašume"
    },
    {
        "title": "Bojana Jelić",
        "full_title": "Fotografija Bojane Jelić, samostalni projektant u Birou za planiranje, PD Srbijašume",
        "image_url": "https://pub-49fb3cc788a74e0a9edbac7e11305b94.r2.dev/Bojana%20Jeli%C4%87.jpg",
        "role": "zaposleni",
        "search_corpus": "fotografija bojane jelić, samostalni projektant u birou za planiranje, pd srbijašume"
    },
    {
        "title": "Boško Malešević",
        "full_title": "Fotografija Boška Maleševića, samostalni projektant u Birou za planiranje, PD Srbijašume",
        "image_url": "https://pub-49fb3cc788a74e0a9edbac7e11305b94.r2.dev/Bo%C5%A1ko%20Male%C5%A1evi%C4%87.jpg",
        "role": "zaposleni",
        "search_corpus": "fotografija boška maleševića, samostalni projektant u birou za planiranje, pd srbijašume"
    },
    {
        "title": "Darko Živanović",
        "full_title": "Fotografija Darka Živanovića, viši projektant u Birou za planiranje, PD Srbijašume",
        "image_url": "https://pub-49fb3cc788a74e0a9edbac7e11305b94.r2.dev/Darko%20%C5%BDivanovi%C4%87.jpg",
        "role": "zaposleni",
        "search_corpus": "fotografija darka živanovića, viši projektant u birou za planiranje, pd srbijašume"
    },
    {
        "title": "Dragana Miladinović",
        "full_title": "Fotografija Dragane Miladinović, samostalni projektant u Birou za planiranje, PD Srbijašume",
        "image_url": "https://pub-49fb3cc788a74e0a9edbac7e11305b94.r2.dev/Dragana%20Miladinovi%C4%87.jpg",
        "role": "zaposleni",
        "search_corpus": "fotografija dragane miladinović, samostalni projektant u birou za planiranje, pd srbijašume"
    },
    {
        "title": "Marina Đukelić",
        "full_title": "Fotografija Marine Đukelić, samostalni projektant u Birou za planiranje, PD Srbijašume",
        "image_url": "https://pub-49fb3cc788a74e0a9edbac7e11305b94.r2.dev/Marina%20D%C5%BEukeli%C4%87.jpg",
        "role": "zaposleni",
        "search_corpus": "fotografija marine đukelić, samostalni projektant u birou za planiranje, pd srbijašume"
    },
    {
        "title": "Mirko Kovačević",
        "full_title": "Fotografija Mirka Kovačevića, viši projektant u Birou za planiranje, PD Srbijašume",
        "image_url": "https://pub-49fb3cc788a74e0a9edbac7e11305b94.r2.dev/Mirko%20Kova%C4%8Devi%C4%87.jpg",
        "role": "zaposleni",
        "search_corpus": "fotografija mirka kovačevića, viši projektant u birou za planiranje, pd srbijašume"
    },
    {
        "title": "Mirko Simonović",
        "full_title": "Fotografija Mirka Simonovića, samostalni projektant u Birou za planiranje, PD Srbijašume",
        "image_url": "https://pub-49fb3cc788a74e0a9edbac7e11305b94.r2.dev/Mirko%20Simonovi%C4%87.jpg",
        "role": "zaposleni",
        "search_corpus": "fotografija mirka simonovića, samostalni projektant u birou za planiranje, pd srbijašume"
    },
    {
        "title": "Nebojša Ivošević",
        "full_title": "Fotografija Nebojše Ivoševića, samostalni projektant u Birou za planiranje, PD Srbijašume",
        "image_url": "https://pub-49fb3cc788a74e0a9edbac7e11305b94.r2.dev/Neboj%C5%A1a%20Ivo%C5%A1evi%C4%87.jpg",
        "role": "zaposleni",
        "search_corpus": "fotografija nebojše ivoševića, samostalni projektant u birou za planiranje, pd srbijašume"
    },
    {
        "title": "Nenad Vamović",
        "full_title": "Fotografija Nenada Vamovića, viši projektant u Birou za planiranje, PD Srbijašume",
        "image_url": "https://pub-49fb3cc788a74e0a9edbac7e11305b94.r2.dev/Nenad%20Vamovi%C4%87.jpg",
        "role": "zaposleni",
        "search_corpus": "fotografija nenada vamovića, viši projektant u birou za planiranje, pd srbijašume"
    },
    {
        "title": "Nenad Vereš",
        "full_title": "Fotografija Nenada Vereša, samostalni projektant u Birou za planiranje, PD Srbijašume",
        "image_url": "https://pub-49fb3cc788a74e0a9edbac7e11305b94.r2.dev/Nenad%20Vere%C5%A1.jpg",
        "role": "zaposleni",
        "search_corpus": "fotografija nenada vereša, samostalni projektant u birou za planiranje, pd srbijašume"
    },
    {
        "title": "Predrag Dedijer",
        "full_title": "Fotografija Predraga Dedijera, samostalni projektant u Birou za planiranje, PD Srbijašume",
        "image_url": "https://pub-49fb3cc788a74e0a9edbac7e11305b94.r2.dev/Predrag%20Dedijer.jpg",
        "role": "zaposleni",
        "search_corpus": "fotografija predraga dedijera, samostalni projektant u birou za planiranje, pd srbijašume"
    },
    {
        "title": "Radoje Ščekić",
        "full_title": "Fotografija Radoja Ščekića, viši projektant u Birou za planiranje, PD Srbijašume",
        "image_url": "https://pub-49fb3cc788a74e0a9edbac7e11305b94.r2.dev/Radoje%20%C5%A0%C4%8Deki%C4%87.jpg",
        "role": "zaposleni",
        "search_corpus": "fotografija radoja ščekića, viši projektant u birou za planiranje, pd srbijašume"
    },
    {
        "title": "Saša Perduh",
        "full_title": "Fotografija Saše Perduha, samostalni projektant u Birou za planiranje, PD Srbijašume",
        "image_url": "https://pub-49fb3cc788a74e0a9edbac7e11305b94.r2.dev/Sa%C5%A1a%20Perduh.jpg",
        "role": "zaposleni",
        "search_corpus": "fotografija saše perduha, samostalni projektant u birou za planiranje, pd srbijašume"
    },
    {
        "title": "Snežana Dubovac",
        "full_title": "Fotografija Snežane Dubovac, geodete u Birou za planiranje, PD Srbijašume",
        "image_url": "https://pub-49fb3cc788a74e0a9edbac7e11305b94.r2.dev/Sne%C5%BEana%20Dubovac.jpg",
        "role": "zaposleni",
        "search_corpus": "fotografija snežane dubovac, geodete u birou za planiranje, pd srbijašume"
    },
    {
        "title": "Vedrana Miljković",
        "full_title": "Fotografija Vedrane Miljković, samostalni projektant u Birou za planiranje, PD Srbijašume",
        "image_url": "https://pub-49fb3cc788a74e0a9edbac7e11305b94.r2.dev/Vedrana%20Miljkovi%C4%87.jpg",
        "role": "zaposleni",
        "search_corpus": "fotografija vedrane miljković, samostalni projektant u birou za planiranje, pd srbijašume"
    },
    {
        "title": "Vladimir Milovanović",
        "full_title": "Fotografija Vladimira Milovanovića, radnik u računovodstvu u Birou za planiranje, PD Srbijašume",
        "image_url": "https://pub-49fb3cc788a74e0a9edbac7e11305b94.r2.dev/Vladimir%20Milovanovi%C4%87.jpg",
        "role": "zaposleni",
        "search_corpus": "fotografija vladimira milovanovića, radnik u računovodstvu u birou za planiranje, pd srbijašume"
    },
    {
        "title": "Vuk Čeperković",
        "full_title": "Fotografija Vuka Čeperkovića, samostalni projektant u Birou za planiranje, PD Srbijašume",
        "image_url": "https://pub-49fb3cc788a74e0a9edbac7e11305b94.r2.dev/Vuk%20%C4%8Ceperkovi%C4%87.jpg",
        "role": "zaposleni",
        "search_corpus": "fotografija vuka čeperkovića, samostalni projektant u birou za planiranje, pd srbijašume"
    },
    {
        "title": "Zoran Petrović",
        "full_title": "Fotografija Zorana Petrovića, viši projektant u Birou za planiranje, PD Srbijašume",
        "image_url": "https://pub-49fb3cc788a74e0a9edbac7e11305b94.r2.dev/Zoran%20Petrovi%C4%87.jpg",
        "role": "zaposleni",
        "search_corpus": "fotografija zorana petrovića, viši projektant u birou za planiranje, pd srbijašume"
    },
    {
        "title": "Čedo Vuković",
        "full_title": "Fotografija Čede Vukovića, samostalni projektant u Birou za planiranje, PD Srbijašume",
        "image_url": "https://pub-49fb3cc788a74e0a9edbac7e11305b94.r2.dev/%C4%8Cedo%20Vukovi%C4%87.jpg",
        "role": "zaposleni",
        "search_corpus": "fotografija čede vukovića, samostalni projektant u birou za planiranje, pd srbijašume"
    }
]

def normalize_text(text: str) -> str:
    if not text: return ""
    text = text.lower().replace("đ", "dj")
    nfkd = unicodedata.normalize('NFKD', text)
    return "".join([c for c in nfkd if not unicodedata.combining(c)])

# ==========================================
# 3. ZAGLAVLJE I SIDEBAR
# ==========================================
col_l1, col_l2, col_l3 = st.columns([1, 4, 1])
with col_l1:
    if SYSTEM_LOGOS["biro"]:
        st.image(SYSTEM_LOGOS["biro"], use_container_width=True)
with col_l2:
    st.markdown("<h1 class='title-text' style='text-align: center;'>🌲 BiroChat</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #2e7d32; font-size: 1.2rem; font-weight: bold;'>Korporativni Asistent</p>", unsafe_allow_html=True)
with col_l3:
    if SYSTEM_LOGOS["srbijasume"]:
        st.image(SYSTEM_LOGOS["srbijasume"], use_container_width=True)

st.markdown("---")

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
    """)

# ==========================================
# 4. CHAT I LOGIKA PRETRAGE
# ==========================================
if "messages" not in st.session_state:
    st.session_state.messages = []

PERSONNEL_KEYWORDS = ["direktor", "direktora", "zamenik", "zamenika", "zamenici", "zaposlen", "zaposleni", "zaposlenih", "radnik", "radnici", "uprava", "spisak"]

def is_personnel_query(question: str) -> bool:
    q_norm = normalize_text(question)
    return any(re.search(r'\b' + re.escape(kw) + r'\b', q_norm) for kw in PERSONNEL_KEYWORDS)

def filter_personnel(catalog, query: str):
    q_norm = normalize_text(query)
    
    # 1. Direktor
    if "direktor" in q_norm and not any(k in q_norm for k in ["zamenik", "zamenika", "zamenici"]):
        res = [p for p in catalog if p["role"] == "direktor"]
        if res: return res

    # 2. Zamenici
    if any(k in q_norm for k in ["zamenik", "zamenika", "zamenici", "goran", "caldovic", "svetlana", "mihajlovic"]):
        res = [p for p in catalog if p["role"] == "zamenik"]
        if res: return res

    # 3. Spisak zaposlenih (vraća sortirani katalog)
    if any(k in q_norm for k in ["zaposlen", "spisak", "radnik", "radnici", "uprava"]):
        return catalog

    # 4. Pretraga po imenu
    query_words = [w for w in q_norm.split() if len(w) > 2 and w not in ["ko", "je", "su", "u", "biro", "biroa", "prikazi", "sliku", "slika", "zaposlenih", "zaposleni"]]
    matched = [p for p in catalog if any(word in p["search_corpus"] for word in query_words)]
    
    return matched if matched else catalog

st.markdown("##### 💡 Brza pitanja:")
quick_questions = [
    "Ko je direktor Biroa?",
    "Ko su zamenici direktora?",
    "Spisak zaposlenih",
    "Koji štampači se koriste u Birou?",
    "Spisak opreme i tonera",
    "Ko je ministar zdravstva u Srbiji?"
]

col1, col2, col3 = st.columns(3)
q_cols = [col1, col2, col3, col1, col2, col3]

selected_quick_q = None
for idx, q in enumerate(quick_questions):
    if q_cols[idx].button(q, key=f"quick_btn_{idx}", use_container_width=True):
        selected_quick_q = q

# Prikaz istorije poruka
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if "content" in msg and msg["content"]:
            st.markdown(msg["content"])
        if "images" in msg and msg["images"]:
            imgs = msg["images"]
            if len(imgs) == 1:
                st.image(imgs[0]["image_url"], caption=imgs[0]["full_title"], use_container_width=True)
            else:
                grid_cols = st.columns(3)
                for idx, img_item in enumerate(imgs):
                    with grid_cols[idx % 3]:
                        st.image(img_item["image_url"], caption=img_item["title"], use_container_width=True)

chat_input_val = st.chat_input("Postavite pitanje o dokumentima ili zaposlenima...")
user_input = selected_quick_q or chat_input_val

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        if is_personnel_query(user_input):
            st.info("🔍 Učitavam registar zaposlenih sa fotografijama...")
            
            role_order = {"direktor": 0, "zamenik": 1, "zaposleni": 2}
            sorted_catalog = sorted(PERSONNEL_CATALOG, key=lambda x: (role_order[x["role"]], x["title"]))
            
            filtered_photos = filter_personnel(sorted_catalog, user_input)
            if filtered_photos:
                answer_text = f"Pronađeno u registru zaposlenih Biroa ({len(filtered_photos)} zaposlenih):"
                st.markdown(answer_text)
                
                # Prikaz slika: ako je 1 ili 2 (direktor/zamenici), prikaži veće; ako je spisak, prikaži u 3 kolone (grid)
                if len(filtered_photos) <= 2:
                    for img in filtered_photos:
                        st.image(img["image_url"], caption=img["full_title"], use_container_width=True)
                else:
                    grid_cols = st.columns(3)
                    for idx, img in enumerate(filtered_photos):
                        with grid_cols[idx % 3]:
                            st.image(img["image_url"], caption=img["title"], use_container_width=True)
                
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer_text,
                    "images": filtered_photos
                })
            else:
                answer_text = "⚠️ U registru nisu pronađeni odgovarajući zaposleni."
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