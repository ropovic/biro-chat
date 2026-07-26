import streamlit as st
from groq import Groq
from qdrant_client import QdrantClient
from fastembed import TextEmbedding

# -------------------------------------------------------------
# 1. KONFIGURACIJA STRANICE
# -------------------------------------------------------------
st.set_page_config(
    page_title="BiroChat Asistent",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 BiroChat Asistent")
st.caption("Postavite pitanje u vezi sa dokumentacijom, tabelama ili zaposlenima.")

# -------------------------------------------------------------
# 2. INICIJALIZACIJA KLIJENATA I MODELA (SA CACHE-OM)
# -------------------------------------------------------------
@st.cache_resource
def init_services():
    # Provera da li su definisane tajne u Streamlit-u
    potrebne_tajne = ["GROQ_API_KEY", "QDRANT_URL", "QDRANT_API_KEY"]
    for tajna in potrebne_tajne:
        if tajna not in st.secrets:
            st.error(f"Nedostaje ključ '{tajna}' u Streamlit Secrets-u!")
            st.stop()
            
    groq_client = Groq(api_key=st.secrets["GROQ_API_KEY"])
    qdrant_client = QdrantClient(
        url=st.secrets["QDRANT_URL"],
        api_key=st.secrets["QDRANT_API_KEY"]
    )
    embedding_model = TextEmbedding()
    return groq_client, qdrant_client, embedding_model

groq_client, qdrant_client, embedding_model = init_services()
KOLEKCIJA_NAZIV = "biro_dokumentacija"

# -------------------------------------------------------------
# 3. FUNKCIJA ZA PRETRAGU QDRANT BAZE
# -------------------------------------------------------------
def pretrazi_bazu(upit_korisnika, top_k=5):
    try:
        # Generisanje vektora za upit korisnika
        vector = list(embedding_model.embed([upit_korisnika]))[0]
        
        # Pretraga u Qdrant bazi
        rezultati = qdrant_client.search(
            collection_name=KOLEKCIJA_NAZIV,
            query_vector=vector.tolist(),
            limit=top_k
        )
        
        kontekst_delovi = []
        for res in rezultati:
            payload = res.payload
            izvor = payload.get("izvor", "Nepoznat izvor")
            tekst = payload.get("tekst", "")
            
            deo = f"--- Izvor: {izvor} ---\n{tekst}"
            
            # Pretraga R2 URL linkova u payload-u (iz Fotobaza.csv)
            for url_key in ["url", "URL", "url_slike", "slika_url", "link", "R2_URL"]:
                if url_key in payload and payload[url_key]:
                    deo += f"\n[SLIKA/LOGO R2 URL]: {payload[url_key]}"
                    break
                    
            kontekst_delovi.append(deo)
            
        return "\n\n".join(kontekst_delovi)
    except Exception as e:
        st.error(f"Greška pri pretrazi Qdrant baze: {e}")
        return ""

# -------------------------------------------------------------
# 4. INICIJALIZACIJA I PRIKAZ ISTORIJE PORUKA
# -------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Zdravo! Ja sam vaš BiroChat asistent. Kako vam mogu pomoći danas?"}
    ]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# -------------------------------------------------------------
# 5. OBRADA UNOSA I GENERISANJE ODGOVORA (GROQ + RAG)
# -------------------------------------------------------------
if upit := st.chat_input("Postavite pitanje..."):
    # Dodaj i prikaži korisničku poruku
    st.session_state.messages.append({"role": "user", "content": upit})
    with st.chat_message("user"):
        st.markdown(upit)

    # Generiši odgovor asistenta
    with st.chat_message("assistant"):
        with st.spinner("Pretražujem bazu i formiram odgovor..."):
            kontekst = pretrazi_bazu(upit)
            
            sistemske_instrukcije = f"""Ti si stručni asistent Biroa. Odgovaraj tačno, profesionalno i na srpskom jeziku.
Koristi ISKLJUČIVO priloženi kontekst iz dokumentacije kako bi odgovorio na pitanje.
Ako tražena informacija ne postoji u kontekstu, pošteno reci da je nemaš u bazi.

VAŽNO PRAVILO ZA FOTOGRAFIJE I LOGO-E:
Ako kontekst sadrži [SLIKA/LOGO R2 URL] ili link ka slici zaposlenog ili logotipu, OBAVEZNO prikaži sliku u svom odgovoru koristeći standardni Markdown format za slike:
![Opis slike](R2_URL)

KONTEKST IZ BAZE DOKUMENTACIJE:
{kontekst}
"""

            try:
                # Poziv Groq API-ja sa strimovanjem odgovora
                stream = groq_client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[
                        {"role": "system", "content": sistemske_instrukcije},
                        {"role": "user", "content": upit}
                    ],
                    temperature=0.2,
                    max_tokens=1024,
                    stream=True
                )
                
                # Strimovanje teksta direktno u Streamlit UI
                odgovor = st.write_stream(stream)
                
                # Sačuvaj odgovor u istoriju chata
                st.session_state.messages.append({"role": "assistant", "content": odgovor})
                
            except Exception as e:
                st.error(f"Greška pri komunikaciji sa Groq API-jem: {e}")