import streamlit as st
from qdrant_client import QdrantClient
from groq import Groq
from fastembed import TextEmbedding

# ----------------- UČITAVANJE KLJUČEVA -----------------
QDRANT_URL = st.secrets["QDRANT_URL"]
QDRANT_API_KEY = st.secrets["QDRANT_API_KEY"]
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]

COLLECTION_NAME = "baza_cloud_v2"

st.set_page_config(page_title="Biro Chat", page_icon="🌲")

# ----------------- INICIJALIZACIJA KLIJENATA -----------------
@st.cache_resource
def init_clients():
    qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    groq = Groq(api_key=GROQ_API_KEY)
    embed_model = TextEmbedding(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    return qdrant, groq, embed_model

qdrant, groq, embed_model = init_clients()

# ----------------- CHAT INTERFEJS -----------------
st.title("🌲 Biro za planiranje — Asistent")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Postavite pitanje o zaposlenima ili Birou..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Pretražujem bazu podataka..."):
            try:
                # 1. Generisanje vektora besplatnim modelom
                query_vector = list(embed_model.embed([prompt]))[0].tolist()

                # 2. Pretraga u Qdrant bazi
                search_response = qdrant.query_points(
                    collection_name=COLLECTION_NAME,
                    query=query_vector,
                    limit=5
                )

                kontekst = "\n".join([hit.payload["tekst"] for hit in search_response.points])

                system_prompt = (
                    "Ti si koristan asistent Biroa za planiranje (PD Srbijašume). "
                    "Odgovaraj tačno na osnovu datog konteksta. "
                    "Ako kontekst sadrži URL fotografije, obavezno ga prikaži korisniku."
                )

                user_prompt = f"Kontekst:\n{kontekst}\n\nPitanje: {prompt}"

                # 3. Slanje na Groq LLM
                response = groq.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.2
                )

                odgovor = response.choices[0].message.content
                st.markdown(odgovor)
                st.session_state.messages.append({"role": "assistant", "content": odgovor})

            except Exception as e:
                st.error(f"Došlo je do greške: {e}")