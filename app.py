import os
import streamlit as st

# ==========================================
# 1. KONFIGURACIJA I PREFIKSI ZA R2 BUCKET
# ==========================================
st.set_page_config(page_title="BiroChat Assistant", page_icon="🤖", layout="wide")

# Separacija R2 prefiksa - ključno za izbegavanje mešanja
R2_PREFIX_EMPLOYEES = "zaposleni/"
R2_PREFIX_DIAGRAMS = "dokumenti_diagrami/"

QUICK_PROMPTS = [
    "Ko su zamenici direktora?",
    "Prikaži važeći pravilnik o radu",
    "Pretraga po delovodnom broju",
    "Organizaciona šema sektora"
]

# ==========================================
# 2. INICIJALIZACIJA SESSION STATE-A
# ==========================================
if "messages" not in st.session_state:
    st.session_state.messages = []

if "pending_prompt" not in st.session_state:
    st.session_state.pending_prompt = None


# ==========================================
# 3. MOCK / REAL R2 MEDIA PRETRAGA SA SEPARACIJOM
# ==========================================
def fetch_r2_media(query_text: str, category: str):
    """
    Pretražuje R2 bucket isključivo u okviru zadate kategorije/foldera.
    category: 'zaposleni' ili 'dijagrami'
    """
    prefix = R2_PREFIX_EMPLOYEES if category == "zaposleni" else R2_PREFIX_DIAGRAMS
    
    # OVDJE INTEGRISATI TVOJ R2 BOTO3 / CLOUDFLARE SDK POZIV:
    # example: response = r2_client.list_objects_v2(Bucket=BUCKET_NAME, Prefix=prefix)
    
    # Trenutni demontracioni fallback pošto je R2 bez odgovarajućih slika:
    return None  # Vraća URL slike ako postoji, inače None


# ==========================================
# 4. LOGIKA ROUTINGA I DETEKCIJE NAMERE (INTENT)
# ==========================================
def classify_intent(query_text: str) -> str:
    """
    Klasifikuje upit u jednu od 3 kategorije:
    1. 'text_only' - Pitanja o funkciji, pravilnicima, zamenicima (Qdrant)
    2. 'employee_photo' - Zahtev za fotografijom određene osobe/radnika
    3. 'technical_media' - Dijagrami, mape, ruže vetrova, skice
    """
    q = query_text.lower()
    
    # Reči koje traže eksplicitno sliku lica / radnika
    employee_photo_kw = ["slika zaposlenog", "fotografija zamenika", "portret", "slika direktora", "profilna slika"]
    
    # Reči koje traže tehničke crteže / grafike
    technical_media_kw = ["dijagram", "šema", "sektorski prikaz", "mapa", "nacrt", "grafik"]

    if any(kw in q for kw in employee_photo_kw):
        return "employee_photo"
    elif any(kw in q for kw in technical_media_kw):
        return "technical_media"
    else:
        # Podrazumevani put za tekstualna pitanja ("Ko su zamenici", "Pravilnik", itd.)
        return "text_only"


def process_user_query(query_text: str):
    """
    Glavni procesor koji izvršava pretragu na osnovu klasifikovane namere.
    """
    intent = classify_intent(query_text)
    
    # CASE 1: EKSPLICITAN ZAHTEV ZA SLIKOM ZAPOSLENOG
    if intent == "employee_photo":
        image_url = fetch_r2_media(query_text, category="zaposleni")
        if image_url:
            return {"type": "image", "content": image_url, "caption": f"Fotografija: {query_text}"}
        else:
            return {
                "type": "text", 
                "content": "⚠️ U bazi zaposlenih (`r2/zaposleni/`) nije pronađena službena fotografija za traženi upit."
            }

    # CASE 2: TEHNIČKI DIJAGRAMI I GRAFIKI
    elif intent == "technical_media":
        image_url = fetch_r2_media(query_text, category="dijagrami")
        if image_url:
            return {"type": "image", "content": image_url, "caption": f"Dijagram: {query_text}"}
        else:
            return {
                "type": "text", 
                "content": "⚠️ U medijskom registru (`r2/dokumenti_diagrami/`) nisu pronađeni odgovarajući dijagrami."
            }

    # CASE 3: TEKSTUALNA PRETRAGA (Qdrant / LlamaIndex)
    else:
        try:
            # Ovde ide poziv ka Qdrant-u / LlamaIndex-u
            # response = query_engine.query(query_text)
            # text_response = str(response)
            
            # Ilustrativan tekstualni odgovor koji odvojen od slika:
            text_response = (
                f"**Rezultat pretrage sistematizacije i odluka za:** *'{query_text}'*\n\n"
                "Na osnovu važeće sistematizacije radnih mesta:\n"
                "- **Zamenik generalnog direktora:** [Ime i Prezime]\n"
                "- **Zamenik direktora tehničkog sektora:** [Ime i Prezime]\n"
                "- **Zamenik direktora za pravne poslove:** [Ime i Prezime]"
            )
            return {"type": "text", "content": text_response}
        except Exception as e:
            return {"type": "text", "content": f"Došlo je do greške prilikom pretrage tekstualne baze: {str(e)}"}


# ==========================================
# 5. RENDEROVANJE INTERFEJSA (STREAMLIT)
# ==========================================
st.title("🤖 BiroChat")
st.caption("Interni pretraživač dokumenata, sistematizacije i medija")

# Prikaz istorije poruka
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if isinstance(msg["content"], dict):
            if msg["content"].get("type") == "image":
                st.image(msg["content"]["content"], caption=msg["content"].get("caption"))
            else:
                st.markdown(msg["content"]["content"])
        else:
            st.markdown(msg["content"])

# --- BRZI UPITI (DUGMIĆI) ---
st.write("---")
st.caption("Brzi upiti:")
cols = st.columns(len(QUICK_PROMPTS))

for idx, prompt_text in enumerate(QUICK_PROMPTS):
    if cols[idx].button(prompt_text, key=f"btn_{idx}", use_container_width=True):
        st.session_state.pending_prompt = prompt_text
        st.rerun()

# --- CHAT INPUT & OBRADA ---
user_input = st.chat_input("Postavite pitanje...")

if st.session_state.pending_prompt:
    active_query = st.session_state.pending_prompt
    st.session_state.pending_prompt = None
elif user_input:
    active_query = user_input
else:
    active_query = None

if active_query:
    # 1. Prikaz pitanja korisnika
    st.session_state.messages.append({"role": "user", "content": active_query})
    with st.chat_message("user"):
        st.markdown(active_query)

    # 2. Generisanje i prikaz odgovora
    with st.chat_message("assistant"):
        with st.spinner("Pretražujem bazu podataka..."):
            result = process_user_query(active_query)
            
            if result["type"] == "image":
                st.image(result["content"], caption=result.get("caption"))
            else:
                st.markdown(result["content"])
            
    st.session_state.messages.append({"role": "assistant", "content": result})