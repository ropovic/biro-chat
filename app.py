import streamlit as st
import re
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchText
from groq import Groq
from fastembed import TextEmbedding

# ----------------- UČITAVANJE KLJUČEVA -----------------
QDRANT_URL = st.secrets["QDRANT_URL"]
QDRANT_API_KEY = st.secrets["QDRANT_API_KEY"]
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]

COLLECTION_NAME = "baza_cloud_v2"

# ----------------- KONFIGURACIJA STRANICE -----------------
st.set_page_config(
    page_title="Biro Chat Asistent",
    page_icon="🌲",
    layout="centered"
)

# ----------------- CUSTOM CSS DIZAJN -----------------
st.markdown("""
    <style>
    .main-title {
        color: #1b4332;
        font-weight: 700;
        margin-bottom: 0px;
        font-size: 2.2rem;
    }
    .sub-title {
        color: #555;
        font-size: 0.95rem;
        margin-bottom: 10px;
    }
    .stChatMessage {
        border-radius: 12px;
        padding: 10px;
        margin-bottom: 8px;
    }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# ----------------- INICIJALIZACIJA KLIJENATA -----------------
@st.cache_resource
def init_clients():
    qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, check_compatibility=False)
    groq = Groq(api_key=GROQ_API_KEY)
    embed_model = TextEmbedding(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    return qdrant, groq, embed_model

qdrant, groq, embed_model = init_clients()

# ----------------- POMOĆNE FUNKCIJE ZA SRPSKI JEZIK -----------------
def ukloni_dijakritike(tekst):
    sve = str.maketrans("čćšđžČĆŠĐŽ", "ccsdzCCSDZ")
    return tekst.translate(sve)

def generisi_korene(rec):
    """Generiše koren reči skidanjem srpskih padežnih nastavaka i dijakritika."""
    r = rec.strip("?,.!\"':;()[]{}").lower()
    if len(r) <= 3:
        return []

    varijacije = {r, ukloni_dijakritike(r)}
    nastavci = ["ovima", "evima", "ama", "ima", "om", "em", "ov", "ev", "in", "og", "u", "a", "e", "i", "o"]

    for nastavak in nastavci:
        if r.endswith(nastavak) and len(r) - len(nastavak) >= 3:
            koren = r[:-len(nastavak)]
            varijacije.add(koren)
            varijacije.add(ukloni_dijakritike(koren))

    return list(varijacije)

# ----------------- FUNKCIJA ZA HIBRIDNU PRETRAGU -----------------
def dobij_hibridni_kontekst(upit):
    rezultujuci_tekstovi = []
    
    # 1. Vektorska pretraga (po opštem smislu)
    query_vector = list(embed_model.embed([upit]))[0].tolist()
    vector_response = qdrant.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=15
    )
    for hit in vector_response.points:
        rezultujuci_tekstovi.append(hit.payload["tekst"])

    # 2. Pametna tekstualna pretraga po korenima reči (rešava padeže i kvačice)
    reci = [w for w in upit.split() if len(w) > 3]
    stop_reci = {"bazi", "bazu", "neki", "neka", "neko", "postoji", "ima", "ovom", "znas", "kazi", "pise", "izvesni", "izvesnog"}

    pretrazeni_koreni = set()
    for rec in reci:
        if rec.lower() in stop_reci:
            continue
        
        koreni = generisi_korene(rec)
        for koren in koreni:
            if koren in pretrazeni_koreni or len(koren) < 3:
                continue
            pretrazeni_koreni.add(koren)

            try:
                kw_hits, _ = qdrant.scroll(
                    collection_name=COLLECTION_NAME,
                    scroll_filter=Filter(
                        must=[FieldCondition(key="tekst", match=MatchText(text=koren))]
                    ),
                    limit=5
                )
                for hit in kw_hits:
                    txt = hit.payload["tekst"]
                    if txt not in rezultujuci_tekstovi:
                        rezultujuci_tekstovi.append(txt)
            except Exception:
                pass

    return "\n\n".join(rezultujuci_tekstovi)

# ----------------- BOČNI MENI (SIDEBAR) -----------------
with st.sidebar:
    st.image("https://pub-49fb3cc788a74e0a9edbac7e11305b94.r2.dev/biro_logo.jpg", use_container_width=True)
    st.title("🌲 Biro Chat")
    st.markdown("**Digitalni asistent Biroa za planiranje**\n\n*PD Srbijašume*")
    st.divider()
    
    st.markdown("### 🛠️ Status sistema")
    st.caption("🟢 **Vektorska baza:** Qdrant Cloud")
    st.caption("🟢 **LLM:** Llama-3.3 (Groq)")
    st.caption("🟢 **Embeddings:** FastEmbed (Local)")
    
    st.divider()
    
    if st.button("🧹 Obriši razgovor", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ----------------- GLAVNO ZAGLAVLJE -----------------
col_logo, col_title = st.columns([1, 4])

with col_logo:
    st.image("https://pub-49fb3cc788a74e0a9edbac7e11305b94.r2.dev/srbijasume_logo.jpg", width=90)

with col_title:
    st.markdown("<h1 class='main-title'>Biro za planiranje</h1>", unsafe_allow_html=True)
    st.markdown("<p class='sub-title'>PD Srbijašume — Digitalni asistent</p>", unsafe_allow_html=True)

st.divider()

if "messages" not in st.session_state:
    st.session_state.messages = []

# ----------------- TRAJNA BRZA PITANJA (EXPANDER) -----------------
with st.expander("💡 Brza predložena pitanja (kliknite da postavite)", expanded=(len(st.session_state.messages) == 0)):
    col1, col2, col3 = st.columns(3)
    clicked_prompt = None
    if col1.button("👔 Ko je direktor?", use_container_width=True):
        clicked_prompt = "Ko je direktor Biroa i pokaži njegovu sliku?"
    if col2.button("👥 Ko su zamenici?", use_container_width=True):
        clicked_prompt = "Ko su zamenici direktora u Birou?"
    if col3.button("🌲 Donji Pek?", use_container_width=True):
        clicked_prompt = "Postoji li Donji pek u bazi i šta piše o njemu?"
        
    if clicked_prompt:
        st.session_state.prompt_input = clicked_prompt

# ----------------- PRIKAZ ISTORIJE PORUKA -----------------
for msg in st.session_state.messages:
    avatar = "👤" if msg["role"] == "user" else "🌲"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])

# ----------------- OBRADA UNOSA KORISNIKA -----------------
prompt = st.chat_input("Postavite pitanje...")

if "prompt_input" in st.session_state and st.session_state.prompt_input:
    prompt = st.session_state.prompt_input
    del st.session_state.prompt_input

if prompt:
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar="🌲"):
        with st.spinner("Pretražujem bazu podataka..."):
            try:
                kontekst = dobij_hibridni_kontekst(prompt)

                system_prompt = (
                    "Ti si ljubazan i stručan asistent Biroa za planiranje (PD Srbijašume).\n"
                    "Odgovaraj tačno na osnovu datog konteksta iz baze podataka i dosadašnjeg razgovora.\n\n"
                    "STROGO PRAVILO ZA PISMO:\n"
                    "Pisac odgovora mora koristiti ISKLJUČIVO srpsku latinicu (Gajevicu).\n"
                    "STROGO JE ZABRANJENO mešanje ćiriličnih i latiničnih slova unutar reči ili rečenice.\n"
                    "Prevedi sve ćirilične pojmove iz konteksta u čistu latinicu.\n\n"
                    "STROGO PRAVILO ZA SLIKE:\n"
                    "Ako u kontekstu postoji URL fotografije tražene osobe ili logoa, UVEK je prikaži koristeći Markdown sintaksu za slike:\n"
                    "![Opis slike](URL_slike)\n"
                    "Nikada nemoj ostavljati samo link bez ![...](...).\n\n"
                    f"KONTEKST IZ BAZE PODATAKA:\n{kontekst}"
                )

                messages_for_groq = [{"role": "system", "content": system_prompt}]
                for msg in st.session_state.messages:
                    messages_for_groq.append({"role": msg["role"], "content": msg["content"]})
                messages_for_groq.append({"role": "user", "content": prompt})

                response = groq.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=messages_for_groq,
                    temperature=0.1
                )

                odgovor = response.choices[0].message.content
                st.markdown(odgovor)

                st.session_state.messages.append({"role": "user", "content": prompt})
                st.session_state.messages.append({"role": "assistant", "content": odgovor})

            except Exception as e:
                st.error(f"Došlo je do greške: {e}")