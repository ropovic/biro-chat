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

# ----------------- UNIVERZALNA NORMALIZACIJA TEKSTA -----------------
def sredi_tekst(tekst):
    if not tekst:
        return ""
    
    # Prvo menjamo digrafe
    tekst = tekst.replace('Љ', 'Lj').replace('љ', 'lj').replace('Њ', 'Nj').replace('њ', 'nj').replace('Џ', 'Dž').replace('џ', 'dž')
    
    # Mapa za konverziju svih ćiriličnih i mešanih slova u čistu latinicu
    zamene = {
        'а': 'a', 'б': 'b', 'в': 'v', 'г': 'g', 'д': 'd', 'đ': 'đ', 'ђ': 'đ',
        'е': 'e', 'ж': 'ž', 'з': 'z', 'и': 'i', 'ј': 'j', 'к': 'k', 'л': 'l',
        'м': 'm', 'н': 'n', 'о': 'o', 'п': 'p', 'р': 'r', 'с': 's', 'т': 't',
        'ћ': 'ć', 'у': 'u', 'ф': 'f', 'х': 'h', 'ц': 'c', 'ч': 'č', 'ш': 'š',
        'А': 'A', 'Б': 'B', 'В': 'V', 'Г': 'G', 'Д': 'D', 'Đ': 'Đ', 'Ђ': 'Đ',
        'Е': 'E', 'Ж': 'Ž', 'З': 'Z', 'И': 'I', 'Ј': 'J', 'К': 'K', 'Л': 'L',
        'М': 'M', 'Н': 'N', 'О': 'O', 'П': 'P', 'Р': 'R', 'С': 'S', 'Т': 'T',
        'Ћ': 'Ć', 'У': 'U', 'Ф': 'F', 'Х': 'H', 'Ц': 'C', 'Ч': 'Č', 'Ш': 'Š'
    }
    
    res = [zamene.get(ch, ch) for ch in tekst]
    return "".join(res)

# ----------------- BRZO KEŠIRANJE SVIH ODLOMAKA IZ BAZE -----------------
@st.cache_data(ttl=1800)
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
                # Odlomke odmah normalizujemo u čistu latinicu
                norm_txt = sredi_tekst(r.payload["tekst"])
                sve_tacke.append(norm_txt)
        
        if next_offset is None or len(records) == 0:
            break
            
        offset = next_offset

    return sve_tacke

# ----------------- NAPREDNA PROVERA POGOTKA ČLANA -----------------
def je_pogodjen_clan(tekst, broj):
    norm_txt = tekst.lower()
    br_str = str(broj)

    # 1. Obrazac: clan 4, cl. 4, cl 4, clan: 4, clan. 4, clan 4.
    pat1 = re.compile(rf'\b(clan|cl)\b[\s\.:\-\*#_]*{br_str}\b')
    if pat1.search(norm_txt):
        return True

    # 2. Obrnuti obrazac: 4. clan, 4.cl
    pat2 = re.compile(rf'\b{br_str}\b[\s\.:\-\*#_]*(clan|cl)\b')
    if pat2.search(norm_txt):
        return True

    # 3. Bliski kontakt: reč clan/cl i broj u razmaku do 30 karaktera
    pat3 = re.compile(rf'\b(clan|cl)\b.{{0,30}}\b{br_str}\b')
    if pat3.search(norm_txt):
        return True

    return False

# ----------------- HIBRIDNA PRETRAGA SA DVOSTRUKIM FILTEROM -----------------
def dobij_hibridni_kontekst(upit, max_karaktera=6000):
    prioritetni_tekstovi = []
    vektorski_tekstovi = []
    svi_vidjeni = set()

    sve_poruke = ucitaj_sve_tekstove()
    brojevi = re.findall(r'\b\d+\b', upit)

# 1. SKENIRANJE ZA TAČAN BROJ ČLANA (SA PRIORITETOM NA SRODNE DOKUMENTE)
    if brojevi:
        for br in brojevi:
            for txt in sve_poruke:
                # Ako tražimo kolektivni ugovor, ignorišemo kartone deponovanih potpisa i tabele
                if "kolektivn" in upit.lower() and "karton deponovanih potpisa" in txt.lower():
                    continue

                if je_pogodjen_clan(txt, br) and txt not in svi_vidjeni:
                    svi_vidjeni.add(txt)
                    prioritetni_tekstovi.append(txt)

# 2. FALLBACK: AKO JE U PITANJU UGOVOR A REČ "ČLAN" NIJE DIREKTNO UZ BROJ
    if brojevi and len(prioritetni_tekstovi) == 0 and ("kolektivn" in upit.lower() or "ugovor" in upit.lower()):
        for txt in sve_poruke:
            txt_low = txt.lower()
            if "kolektivn" in txt_low and any(re.search(rf'\b{br}\b', txt_low) for br in brojevi):
                if txt not in svi_vidjeni:
                    svi_vidjeni.add(txt)
                    prioritetni_tekstovi.append(txt)

# 3. VEKTORSKA PRETRAGA ZA ŠIROKI SEMANTIČKI KONTEKST
    norm_upit = sredi_tekst(upit)
    query_vector = list(embed_model.embed([norm_upit]))[0].tolist()
    vector_response = qdrant.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=15
    )
    for hit in vector_response.points:
        raw_txt = hit.payload.get("tekst", "")
        if raw_txt:
            txt = sredi_tekst(raw_txt)
            if txt not in svi_vidjeni:
                svi_vidjeni.add(txt)
                vektorski_tekstovi.append(txt)

    # Spajanje: Prioritetni odlomci idu PRVI
    spojeni_rezultati = prioritetni_tekstovi + vektorski_tekstovi
    spojeni_tekst = "\n\n--- ODLOMAK IZ BAZE ---\n\n".join(spojeni_rezultati)

    if len(spojeni_tekst) > max_karaktera:
        spojeni_tekst = spojeni_tekst[:max_karaktera] + "\n...[Kontekst skraćen radi limita]..."

    return spojeni_tekst, len(prioritetni_tekstovi), len(sve_poruke)

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
                kontekst, br_prioritetnih, ukupno_keširano = dobij_hibridni_kontekst(prompt)

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

                # DEBUG EXPANDER - Prikaz preuzetog konteksta
                with st.expander("🔍 Pregled preuzetog konteksta iz baze (Za debug)"):
                    st.caption(f"Ukupno učitano odlomaka iz Qdrant baze u keš: **{ukupno_keširano}**")
                    st.caption(f"Pronađeno prioritetnih odlomaka sa traženim članom: **{br_prioritetnih}**")
                    st.text_area("Sadržaj poslat Llami:", value=kontekst, height=220)

                st.session_state.messages.append({"role": "user", "content": prompt})
                st.session_state.messages.append({"role": "assistant", "content": odgovor})

            except Exception as e:
                st.error(f"Došlo je do greške: {e}")