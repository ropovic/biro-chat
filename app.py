import streamlit as st
import re
from qdrant_client import QdrantClient
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

# ----------------- BRZO KEŠIRANJE SVIH ODLOMAKA IZ BAZE -----------------
@st.cache_data(ttl=1800)  # Kešira odlomke na 30 minuta
def ucitaj_sve_tekstove():
    sve_tacke = []
    offset = None
    while True:
        records, next_offset = qdrant.scroll(
            collection_name=COLLECTION_NAME,
            limit=250,
            offset=offset,
            with_payload=True,
            with_vectors=False
        )
        for r in records:
            if r.payload and "tekst" in r.payload:
                sve_tacke.append(r.payload["tekst"])
        if next_offset is None or len(records) == 0:
            break
    return sve_tacke

# ----------------- FUNKCIJE ZA KONVERZIJU PISMA -----------------
def cirilica_u_latinicu(tekst):
    digrafi = {'Љ': 'Lj', 'љ': 'lj', 'Њ': 'Nj', 'њ': 'nj', 'Џ': 'Dž', 'џ': 'dž'}
    monografi = {
        'А': 'A', 'а': 'a', 'Б': 'B', 'б': 'b', 'В': 'V', 'в': 'v',
        'Г': 'G', 'г': 'g', 'Д': 'D', 'д': 'd', 'Ђ': 'Đ', 'ђ': 'đ',
        'Е': 'E', 'е': 'e', 'Ж': 'Ž', 'ж': 'ž', 'З': 'Z', 'з': 'z',
        'И': 'I', 'и': 'i', 'Ј': 'J', 'ј': 'j', 'К': 'K', 'к': 'k',
        'Л': 'L', 'л': 'l', 'М': 'M', 'м': 'm', 'Н': 'N', 'н': 'n',
        'О': 'O', 'о': 'o', 'П': 'P', 'п': 'p', 'Р': 'R', 'р': 'r',
        'С': 'S', 'с': 's', 'Т': 'T', 'т': 't', 'Ћ': 'Ć', 'ћ': 'ć',
        'У': 'U', 'у': 'u', 'Ф': 'F', 'ф': 'f', 'Х': 'H', 'х': 'h',
        'Ц': 'C', 'ц': 'c', 'Ч': 'Č', 'ч': 'č', 'Ш': 'Š', 'ш': 'š'
    }
    for k, v in digrafi.items():
        tekst = tekst.replace(k, v)
    res = [monografi.get(ch, ch) for ch in tekst]
    return "".join(res)

# ----------------- HIBRIDNA PRETRAGA SA LOKALNIM REGEX SKENIRANJEM -----------------
def dobij_hibridni_kontekst(upit, max_karaktera=6000):
    prioritetni_tekstovi = []
    vektorski_tekstovi = []
    svi_vidjeni = set()

    # 1. DIREKTNO SKENIRANJE KEŠIRANIH TEKSTOVA ZA BR. ČLANA (100% SIGURNOST)
    brojevi = re.findall(r'\b\d+\b', upit)
    if brojevi:
        sve_poruke = ucitaj_sve_tekstove()
        for br in brojevi:
            # Obrazac hvata: Član 4, Član 4., Члан 4, ЧЛАН 4., čl. 4 itd.
            pattern = re.compile(rf'(?i)\b(član|члан|čl|чл)\.?\s*{br}\b')
            for txt in sve_poruke:
                if pattern.search(txt) and txt not in svi_vidjeni:
                    svi_vidjeni.add(txt)
                    prioritetni_tekstovi.append(txt)

    # 2. VEKTORSKA PRETRAGA ZA ŠIROKI SEMANTIČKI KONTEKST
    query_vector = list(embed_model.embed([upit]))[0].tolist()
    vector_response = qdrant.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=15
    )
    for hit in vector_response.points:
        txt = hit.payload.get("tekst", "")
        if txt and txt not in svi_vidjeni:
            svi_vidjeni.add(txt)
            vektorski_tekstovi.append(txt)

    # Spajanje: Odlomci sa tačnim brojem člana idu PRVI na listi
    spojeni_rezultati = prioritetni_tekstovi + vektorski_tekstovi
    spojeni_tekst = "\n\n--- ODLOMAK IZ BAZE ---\n\n".join(spojeni_rezultati)
    
    # Prevođenje ćirilice u čistiju latinicu pre slanja LLM-u
    spojeni_tekst = cirilica_u_latinicu(spojeni_tekst)

    if len(spojeni_tekst) > max_karaktera:
        spojeni_tekst = spojeni_tekst[:max_karaktera] + "\n...[Kontekst skraćen radi limita]..."

    return spojeni_tekst, len(prioritetni_tekstovi)

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
    
    if st.button("🔄 Osveži keš baze", use_container_width=True):
        st.cache_data.clear()
        st.success("Keš je osvežen!")

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
    col1, col2, col3, col4 = st.columns(4)
    clicked_prompt = None
    if col1.button("👔 Ko je direktor?", use_container_width=True):
        clicked_prompt = "Ko je direktor Biroa i pokaži njegovu sliku?"
    if col2.button("👥 Ko su zamenici?", use_container_width=True):
        clicked_prompt = "Ko su zamenici direktora u Birou?"
    if col3.button("🌲 Crni vrh?", use_container_width=True):
        clicked_prompt = "Postoji li Crni vrh u bazi i šta piše o njemu?"
    if col4.button("📜 Član 4. Kol. ugovora?", use_container_width=True):
        clicked_prompt = "Navedi član 4. kolektivnog ugovora?"

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
                kontekst, br_prioritetnih = dobij_hibridni_kontekst(prompt)

                system_prompt = (
                    "Ti si ljubazan i stručan asistent Biroa za planiranje (PD Srbijašume).\n"
                    "Odgovaraj tačno i direktno na osnovu datog konteksta iz baze podataka i dosadašnjeg razgovora.\n\n"
                    "PRAVILA ODGOVARANJA:\n"
                    "1. Odgovaraj na srpskom jeziku (latinica).\n"
                    "2. Daj direktan i potpun odgovor na postavljeno pitanje.\n"
                    "3. Ako u kontekstu postoji URL fotografije tražene osobe ili logoa, prikaži je koristeći Markdown sintaksu: ![Opis slike](URL_slike).\n\n"
                    f"KONTEKST IZ BAZE PODATAKA:\n{kontekst}"
                )

                messages_for_groq = [{"role": "system", "content": system_prompt}]
                
                skracena_istorija = st.session_state.messages[-6:]
                for msg in skracena_istorija:
                    messages_for_groq.append({"role": msg["role"], "content": msg["content"]})
                
                messages_for_groq.append({"role": "user", "content": prompt})

                response = groq.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=messages_for_groq,
                    temperature=0.1
                )

                odgovor = response.choices[0].message.content
                st.markdown(odgovor)

                # DEBUG EXPANDER - Prikaz onoga što je poslato Llami
                with st.expander("🔍 Pregled preuzetog konteksta iz baze (Za debug)"):
                    st.caption(f"Pronađeno prioritetnih odlomaka sa tačnim brojem člana: **{br_prioritetnih}**")
                    st.text_area("Sadržaj poslat Llami:", value=kontekst, height=200)

                st.session_state.messages.append({"role": "user", "content": prompt})
                st.session_state.messages.append({"role": "assistant", "content": odgovor})

            except Exception as e:
                st.error(f"Došlo je do greške: {e}")