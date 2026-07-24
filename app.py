import streamlit as st
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
    /* Stilizovanje glavnog naslova */
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
    /* Zaobljene ivice za chat poruke */
    .stChatMessage {
        border-radius: 12px;
        padding: 10px;
        margin-bottom: 8px;
    }
    /* Sakrivanje Streamlit watermark-a */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# ----------------- INICIJALIZACIJA KLIJENATA -----------------
@st.cache_resource
def init_clients():
    qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    groq = Groq(api_key=GROQ_API_KEY)
    embed_model = TextEmbedding(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    return qdrant, groq, embed_model

qdrant, groq, embed_model = init_clients()

# ----------------- BOČNI MENI (SIDEBAR) -----------------
with st.sidebar:
    # Logo Biroa u bočnom meniju
    st.image("https://pub-49fb3cc788a74e0a9edbac7e11305b94.r2.dev/biro_logo.jpg", use_container_width=True)
    st.title("🌲 Biro Chat")
    st.markdown("**Digitalni asistent Biroa za planiranje**\n\n*PD Srbijašume*")
    st.divider()
    
    st.markdown("### 🛠️ Status sistema")
    st.caption("🟢 **Vektorska baza:** Qdrant Cloud")
    st.caption("🟢 **LLM:** Llama-3.3 (Groq)")
    st.caption("🟢 **Embeddings:** FastEmbed (Local)")
    
    st.divider()
    
    # Dugme za čišćenje istorije razgovora
    if st.button("🧹 Obriši razgovor", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ----------------- GLAVNO ZAGLAVLJE SA LOGOM SRBIJAŠUMA -----------------
col_logo, col_title = st.columns([1, 4])

with col_logo:
    st.image("https://pub-49fb3cc788a74e0a9edbac7e11305b94.r2.dev/srbijasume_logo.jpg", width=90)

with col_title:
    st.markdown("<h1 class='main-title'>Biro za planiranje</h1>", unsafe_allow_html=True)
    st.markdown("<p class='sub-title'>PD Srbijašume — Digitalni asistent</p>", unsafe_allow_html=True)

st.divider()

if "messages" not in st.session_state:
    st.session_state.messages = []

# ----------------- PREDLOŽENA PITANJA (BRZI DUGMIĆI) -----------------
if len(st.session_state.messages) == 0:
    st.markdown("**Brza pitanja za početak:**")
    col1, col2, col3 = st.columns(3)
    
    clicked_prompt = None
    if col1.button("👔 Ko je direktor?", use_container_width=True):
        clicked_prompt = "Ko je direktor Biroa i pokaži njegovu sliku?"
    if col2.button("👥 Ko su zamenici?", use_container_width=True):
        clicked_prompt = "Ko su zamenici direktora u Birou?"
    if col3.button("💻 Ko radi sa bazom?", use_container_width=True):
        clicked_prompt = "Ko je rukovalac bazom podataka i kako izgleda?"
        
    if clicked_prompt:
        st.session_state.prompt_input = clicked_prompt

# ----------------- PRIKAZ ISTORIJE PORUKA -----------------
for msg in st.session_state.messages:
    avatar = "👤" if msg["role"] == "user" else "🌲"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])

# ----------------- OBRADA UNOSA KORISNIKA -----------------
prompt = st.chat_input("Postavite pitanje...")

# Ako je kliknuto neko od brzih dugmadi
if "prompt_input" in st.session_state and st.session_state.prompt_input:
    prompt = st.session_state.prompt_input
    del st.session_state.prompt_input

if prompt:
    # Prikaz pitanja korisnika
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar="🌲"):
        with st.spinner("Pretražujem bazu podataka..."):
            try:
                # 1. Pametna kontekstualna pretraga (spajanje sa prethodnim pitanjem)
                search_query = prompt
                if len(st.session_state.messages) >= 2:
                    last_user_msg = next((m["content"] for m in reversed(st.session_state.messages) if m["role"] == "user"), "")
                    if last_user_msg:
                        search_query = f"{last_user_msg} {prompt}"

                # 2. Generisanje vektora i pretraga u Qdrant-u (limit=10)
                query_vector = list(embed_model.embed([search_query]))[0].tolist()
                
                search_response = qdrant.query_points(
                    collection_name=COLLECTION_NAME,
                    query=query_vector,
                    limit=10
                )

                kontekst = "\n".join([hit.payload["tekst"] for hit in search_response.points])

                # 3. System prompt
                system_prompt = (
                    "Ti si ljubazan i stručan asistent Biroa za planiranje (PD Srbijašume).\n"
                    "Odgovaraj tačno na osnovu datog konteksta iz baze podataka i dosadašnjeg razgovora.\n\n"
                    "STROGO PRAVILO ZA SLIKE:\n"
                    "Ako u kontekstu postoji URL fotografije, UVEK je prikaži koristeći Markdown sintaksu za slike:\n"
                    "![Opis slike](URL_slike)\n"
                    "Nikada nemoj ostavljati samo link bez ![...](...).\n\n"
                    f"KONTEKST IZ BAZE PODATAKA:\n{kontekst}"
                )

                # 4. Sastavljanje istorije za Groq
                messages_for_groq = [{"role": "system", "content": system_prompt}]
                for msg in st.session_state.messages:
                    messages_for_groq.append({"role": msg["role"], "content": msg["content"]})
                messages_for_groq.append({"role": "user", "content": prompt})

                # 5. Poziv ka LLM-u
                response = groq.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=messages_for_groq,
                    temperature=0.2
                )

                odgovor = response.choices[0].message.content
                st.markdown(odgovor)

                # Čuvanje u istoriju
                st.session_state.messages.append({"role": "user", "content": prompt})
                st.session_state.messages.append({"role": "assistant", "content": odgovor})

            except Exception as e:
                st.error(f"Došlo je do greške: {e}")