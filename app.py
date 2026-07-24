import streamlit as st
from qdrant_client import QdrantClient
from groq import Groq
from openai import OpenAI

# ----------------- BEZBEDNO UČITAVANJE KLJUČEVA -----------------
# Ključevi se čitaju iz .streamlit/secrets.toml (lokalno) 
# ili iz Secrets podešavanja na Streamlit Cloud-u
QDRANT_URL = st.secrets["QDRANT_URL"]
QDRANT_API_KEY = st.secrets["QDRANT_API_KEY"]
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]

COLLECTION_NAME = "baza_cloud_v1"
EMBEDDING_MODEL = "text-embedding-3-small"
# -----------------------------------------------------------------

@st.cache_resource
def init_clients():
    qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    groq = Groq(api_key=GROQ_API_KEY)
    openai_client = OpenAI(api_key=OPENAI_API_KEY)
    return qdrant, groq, openai_client

qdrant_client, groq_client, openai_client = init_clients()

# Funkcija koja na osnovu istorije razgovora preformuliše kratka pitanja (sa "on", "njegov"...) u kontekstualna
def contextualize_query(chat_history, latest_question):
    if not chat_history:
        return latest_question
    
    # Izvlačimo poslednjih par poruka radi konteksta
    history_text = ""
    for msg in chat_history[-4:]:
        role = "Korisnik" if msg["role"] == "user" else "Asistent"
        history_text += f"{role}: {msg['content']}\n"
        
    rewrite_prompt = (
        "Na osnovu istorije razgovora i poslednjeg pitanja korisnika, preformuliši poslednje pitanje "
        "tako da bude potpuno samostalno i precizno na srpskom jeziku (ubaci imena, funkcije ili pojmove na koje se odnosi zamenica 'on', 'njegov', 'to' i slično).\n"
        "VAŽNO: Ne odgovaraj na pitanje, samo vrati preformulisani tekst pitanja!\n\n"
        f"Istorija razgovora:\n{history_text}\n"
        f"Poslednje pitanje: {latest_question}\n"
        "Samostalno pitanje:"
    )
    
    try:
        res = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": rewrite_prompt}],
            temperature=0.0
        )
        standalone_q = res.choices[0].message.content.strip()
        return standalone_q
    except Exception:
        return latest_question

def get_query_embedding(text):
    response = openai_client.embeddings.create(
        input=text,
        model=EMBEDDING_MODEL
    )
    return response.data[0].embedding

def search_qdrant(query_vector):
    # Povećano na limit=5 radi obuhvatnije pretrage
    search_results = qdrant_client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=5
    )
    unique_texts = set()
    for point in search_results.points:
        if point.payload and "text" in point.payload:
            unique_texts.add(point.payload["text"])
    return "\n---\n".join(unique_texts)

# Interfejs
st.set_page_config(page_title="Biro Baza Asistent", page_icon="🏢", layout="centered")

st.markdown("""
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .main-title { font-size: 24px; font-weight: 700; color: #2b3fe0; margin-bottom: 0px; }
    </style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.image("https://img.icons8.com/color/96/organization.png", width=64)
    st.markdown("### Biro za planiranje")
    st.markdown("Sistem u potpunosti hostovan na cloudu.")
    st.divider()
    if st.button("🗑️ Obriši istoriju razgovora", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

st.markdown('<p class="main-title">🗂️ Biro Baza - Cloud Asistent</p>', unsafe_allow_html=True)
st.caption("Postavite pitanje o radnicima ili strukturi preduzeća.")
st.divider()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    avatar = "🧑‍💻" if message["role"] == "user" else "🏢"
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])

if prompt := st.chat_input("Unesite pitanje..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar="🏢"):
        with st.spinner("Pretražujem bazu..."):
            try:
                # 1. Preformulisanje upita na osnovu konteksta razgovora
                smart_prompt = contextualize_query(st.session_state.messages[:-1], prompt)
                
                # 2. Vektorizacija preformulisanog pitanja
                q_vector = get_query_embedding(smart_prompt)
                
                # 3. Pretraga Qdrant-a
                context = search_qdrant(q_vector)
                
                # 4. Generisanje kon konačnog odgovora
                system_prompt = (
                    "Ti si profesionalni asistent isključivo za Biro za planiranje. "
                    "Koristi isključivo prosleđeni kontekst.\n"
                    "PRAVILA:\n"
                    "1. Budi direktan i precizan, bez suvišnih generičkih pohvala i floskula.\n"
                    "2. Obrati pažnju na rod (koristi 'oboje' umesto 'obojica' za mešoviti sastav).\n"
                    "3. Ako se u kontekstu nalazi link ka fotografiji tražene osobe ili predmeta, MORAŠ ga prikazati tačno u ovom formatu: ![Opis](URL)."
                )
                
                user_content = f"Kontekst:\n{context}\n\nPitanje korisnika: {prompt}"
                
                completion = groq_client.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_content}
                    ],
                    temperature=0.1
                )
                
                bot_reply = completion.choices[0].message.content
                st.markdown(bot_reply)
                st.session_state.messages.append({"role": "assistant", "content": bot_reply})
                
            except Exception as e:
                st.error(f"Došlo je do greške: {e}")