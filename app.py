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

# Prikaz prethodnih poruka iz istorije
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Postavite pitanje o zaposlenima ili Birou..."):
    # Prikaz trenutnog pitanja korisnika
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Pretražujem bazu podataka..."):
            try:
                # 1. Pametna pretraga: Spajamo prethodno pitanje sa trenutnim ako postoji zamenica (npr. "njegovu")
                search_query = prompt
                if len(st.session_state.messages) >= 2:
                    last_user_msg = next((m["content"] for m in reversed(st.session_state.messages) if m["role"] == "user"), "")
                    if last_user_msg:
                        search_query = f"{last_user_msg} {prompt}"

                # 2. Generisanje vektora i pretraga u Qdrant bazi
                query_vector = list(embed_model.embed([search_query]))[0].tolist()
                
                search_response = qdrant.query_points(
                    collection_name=COLLECTION_NAME,
                    query=query_vector,
                    limit=10
                )

                kontekst = "\n".join([hit.payload["tekst"] for hit in search_response.points])

                # 3. System prompt sa strogim uputstvom za Markdown slike
                system_prompt = (
                    "Ti si koristan asistent Biroa za planiranje (PD Srbijašume).\n"
                    "Odgovaraj tačno na osnovu datog konteksta iz baze podataka i dosadašnjeg razgovora.\n\n"
                    "STROGO PRAVILO ZA SLIKE:\n"
                    "Ako u kontekstu postoji URL fotografije, UVEK je prikaži koristeći Markdown sintaksu za slike:\n"
                    "![Opis slike](URL_slike)\n"
                    "Primer: ![Brano Vamović](https://pub-49fb3cc788a74e0a9edbac7e11305b94.r2.dev/brano_vamovic.jpg)\n"
                    "Nikada nemoj ostavljati go URL link bez ![...](...).\n\n"
                    f"KONTEKST IZ BAZE PODATAKA:\n{kontekst}"
                )

                # 4. Sastavljanje kompletne istorije konverzacije za Groq
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

                # Sačuvanje u istoriju tek NAKON uspešnog odgovora
                st.session_state.messages.append({"role": "user", "content": prompt})
                st.session_state.messages.append({"role": "assistant", "content": odgovor})

            except Exception as e:
                st.error(f"Došlo je do greške: {e}")