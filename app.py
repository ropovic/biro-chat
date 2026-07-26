import streamlit as st
import re
from qdrant_client import QdrantClient
from groq import Groq
from fastembed import TextEmbedding

# Siguran import za TextRerank
try:
    from fastembed import TextRerank
    HAS_RERANKER = True
except ImportError:
    HAS_RERANKER = False

# ----------------- KONFIGURACIJA STRANICE -----------------
st.set_page_config(
    page_title="Biro Chat Asistent",
    page_icon="🌲",
    layout="centered"
)

# ----------------- CUSTOM CSS DIZAJN -----------------
st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(135deg, #1b4332 0%, #2d6a4f 100%);
        padding: 20px;
        border-radius: 12px;
        color: white;
        text-align: center;
        margin-bottom: 25px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .main-header h1 {
        color: white !important;
        margin: 0;
        font-size: 2.2rem;
        font-weight: 700;
    }
    .main-header p {
        color: #d8f3dc !important;
        margin-top: 5px;
        font-size: 1.05rem;
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

# ----------------- UČITAVANJE KLJUČEVA -----------------
potrebne_tajne = ["QDRANT_URL", "QDRANT_API_KEY", "GROQ_API_KEY"]
for tajna in potrebne_tajne:
    if tajna not in st.secrets:
        st.error(f"❌ Nedostaje ključ '{tajna}' u Streamlit Secrets-u!")
        st.stop()

QDRANT_URL = st.secrets["QDRANT_URL"]
QDRANT_API_KEY = st.secrets["QDRANT_API_KEY"]
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]

COLLECTION_NAME = "baza_cloud_v2"

# ----------------- INICIJALIZACIJA KLIJENATA -----------------
@st.cache_resource
def init_clients():
    qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, check_compatibility=False)
    groq_client = Groq(api_key=GROQ_API_KEY)
    embed_model = TextEmbedding(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    
    reranker_model = None
    if HAS_RERANKER:
        try:
            reranker_model = TextRerank(model_name="BAAI/bge-reranker-base")
        except Exception:
            reranker_model = None
            
    return qdrant, groq_client, embed_model, reranker_model

qdrant, groq_client, embed_model, reranker_model = init_clients()

# ----------------- UNIVERZALNA NORMALIZACIJA TEKSTA -----------------
def sredi_tekst(tekst):
    if not tekst:
        return ""
    
    tekst = str(tekst).replace('Љ', 'Lj').replace('љ', 'lj').replace('Њ', 'Nj').replace('њ', 'nj').replace('Џ', 'Dž').replace('џ', 'dž')
    
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
            if r.payload:
                raw_txt = (r.payload.get("tekst") or r.payload.get("text") or 
                           r.payload.get("content") or r.payload.get("page_content") or "")
                izvor = (r.payload.get("naziv_dokumenta") or r.payload.get("file_name") or 
                         r.payload.get("izvor") or r.payload.get("dokument") or 
                         r.payload.get("source") or "")
                
                slika_url = r.payload.get("slika_url") or ""
                
                if raw_txt:
                    sve_tacke.append({
                        "tekst": sredi_tekst(raw_txt),
                        "izvor": sredi_tekst(izvor),
                        "slika_url": slika_url
                    })
        
        if next_offset is None or len(records) == 0:
            break
            
        offset = next_offset

    return sve_tacke

# ----------------- ROBUSTNA PRETRAGA ČLANA -----------------
def pronadji_tacnan_clan(svi_odlomci, broj_str):
    rezultati = []
    for idx, item in enumerate(svi_odlomci):
        txt = item["tekst"]
        izvor = item["izvor"]
        slika_url = item.get("slika_url", "")
        txt_low = txt.lower()
        
        ima_rec = any(w in txt_low for w in ["član", "clan", "čl", "cl", "члан", "член", "чл"])
        ima_broj = (
            f" {broj_str} " in f" {txt_low} " or 
            f"{broj_str}." in txt_low or 
            f"{broj_str})" in txt_low or 
            f"0{broj_str}" in txt_low or
            f"član {broj_str}" in txt_low or
            f"члан {broj_str}" in txt_low
        )
        
        if ima_rec and ima_broj:
            prosirani_tekst = txt
            for step in range(1, 3):
                if idx + step < len(svi_odlomci):
                    sledeci_item = svi_odlomci[idx + step]
                    prosirani_tekst += "\n" + sledeci_item["tekst"]
            rezultati.append({"tekst": prosirani_tekst, "izvor": izvor, "slika_url": slika_url})
            
    return rezultati

# ----------------- FILTRIRANJE I BODOVANJE KANDIDATA -----------------
def filtriraj_i_skoruj_kandidate(svi_kandidati, upit):
    upit_low = upit.lower()
    
    je_dokument_pitanje = any(w in upit_low for w in ["dokument", "naziv", "fajl", "spisak", "koji dokumenti"])
    je_kyocera = "kyocera" in upit_low or "štampač" in upit_low or "stampac" in upit_low
    je_mrcajevac = "mrčajevac" in upit_low or "mrcajevac" in upit_low
    je_direktor = any(w in upit_low for w in ["direktor", "zamenik", "zamenici", "rukovodstv", "uprava", "sef", "šef"])

    skorovani_kandidati = []

    for item in svi_kandidati:
        txt = item["tekst"]
        izvor = item["izvor"]
        slika_url = item.get("slika_url", "")
        txt_low = txt.lower()
        izvor_low = izvor.lower()
        skor = 10 # Osnovni skor da nijedan kandidat ne bude potpuno ignorisan

        if je_kyocera and ("kyocera" in txt_low or "štampač" in txt_low or "stampac" in txt_low):
            skor += 50000

        if je_mrcajevac and "mrčajevac" in txt_low:
            skor += 50000

        if je_direktor:
            if "zamenik" in txt_low or "direktor" in txt_low:
                skor += 5000
            if "http" in txt_low:
                skor += 5000

        if je_dokument_pitanje and izvor and izvor != "zaposleni_i_foto" and izvor != "osnovne_informacije":
            skor += 10000

        upit_reci = [r for r in upit_low.split() if len(r) > 3]
        for rec in upit_reci:
            if rec in txt_low or rec in izvor_low:
                skor += 50

        skorovani_kandidati.append((skor, txt, izvor, slika_url))

    skorovani_kandidati.sort(key=lambda x: x[0], reverse=True)
    return skorovani_kandidati[:15]

# ----------------- HIBRIDNA PRETRAGA SA SLIKAMA -----------------
def dobij_hibridni_kontekst(upit, top_k_rezultata=6, max_karaktera=4000):
    svi_odlomci = ucitaj_sve_tekstove()
    svi_kandidati = []
    svi_vidjeni = set()
    norm_upit = sredi_tekst(upit)
    upit_low = norm_upit.lower()
    brojevi = re.findall(r'\b\d+\b', upit)
    pronadjene_slike_urls = set()

    if brojevi and any(w in upit_low for w in ["clan", "član", "cl", "čl", "члан", "чл"]):
        for br in brojevi:
            direktni_pogodci = pronadji_tacnan_clan(svi_odlomci, br)
            for dp in direktni_pogodci:
                if dp["tekst"] not in svi_vidjeni:
                    svi_vidjeni.add(dp["tekst"])
                    svi_kandidati.append(dp)

    if any(w in upit_low for w in ["dokument", "naziv", "fajl", "spisak", "koji dokumenti"]):
        jedinstveni_izvori = sorted(list(set(item["izvor"] for item in svi_odlomci if item["izvor"])))
        spisak_tekst = "Dostupni nazivi dokumenata u bazi:\n" + "\n".join([f"- {izv}" for izv in jedinstveni_izvori])
        svi_kandidati.append({"tekst": spisak_tekst, "izvor": "Svi dokumenti", "slika_url": ""})

    query_vector = list(embed_model.embed([norm_upit]))[0].tolist()
    
    if hasattr(qdrant, "query_points"):
        vector_response = qdrant.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            limit=25
        )
        points = vector_response.points
    else:
        vector_response = qdrant.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_vector,
            limit=25
        )
        points = vector_response

    for hit in points:
        if hit.payload:
            raw_txt = (hit.payload.get("tekst") or hit.payload.get("text") or 
                       hit.payload.get("content") or hit.payload.get("page_content") or "")
            izvor = (hit.payload.get("naziv_dokumenta") or hit.payload.get("file_name") or 
                     hit.payload.get("izvor") or hit.payload.get("dokument") or 
                     hit.payload.get("source") or "")
            slika_url = hit.payload.get("slika_url") or ""
            
            if raw_txt:
                norm_txt = sredi_tekst(raw_txt)
                if norm_txt not in svi_vidjeni:
                    svi_vidjeni.add(norm_txt)
                    svi_kandidati.append({
                        "tekst": norm_txt,
                        "izvor": sredi_tekst(izvor),
                        "slika_url": slika_url
                    })

    # Fallback ako vektorska pretraga ne vrati ništa, uzmi prve iz keša da sistem ne pukne
    if not svi_kandidati and svi_odlomci:
        svi_kandidati = svi_odlomci[:20]

    skorovani = filtriraj_i_skoruj_kandidate(svi_kandidati, upit)
    txt_to_url = {item[1]: item[3] for item in skorovani if item[3]}

    top_prioritetni = [item[1] for item in skorovani if item[0] >= 5000]
    ostali_kandidati = [item[1] for item in skorovani if item[0] < 5000]
    top_odlomci = list(top_prioritetni)

    if reranker_model is not None and len(ostali_kandidati) > 0 and len(top_odlomci) < top_k_rezultata:
        try:
            potrebno = top_k_rezultata - len(top_odlomci)
            reranked = list(reranker_model.rerank(query=norm_upit, documents=ostali_kandidati))
            reranked.sort(key=lambda x: x["score"], reverse=True)
            top_odlomci.extend([res["document"] for res in reranked[:potrebno]])
        except Exception:
            pass

    if len(top_odlomci) < top_k_rezultata:
        potrebno = top_k_rezultata - len(top_odlomci)
        top_odlomci.extend(ostali_kandidati[:potrebno])

    kontekst_lista = []
    for txt in top_odlomci:
        if txt.startswith("Izvor") or txt.startswith("Dostupni nazivi"):
            kontekst_lista.append(txt)
        else:
            kontekst_lista.append(f"Odlomak iz baze:\n{txt}")
            
        if txt in txt_to_url:
            pronadjene_slike_urls.add(txt_to_url[txt])

    spojeni_tekst = "\n\n--- ODLOMAK IZ BAZE ---\n\n".join(kontekst_lista)

    if len(spojeni_tekst) > max_karaktera:
        spojeni_tekst = spojeni_tekst[:max_karaktera] + "\n...[Kontekst skraćen radi limita]..."

    return spojeni_tekst, len(skorovani), len(svi_odlomci), list(pronadjene_slike_urls)

# ----------------- STRIMOVANJE GROQ ODGOVORA -----------------
def strimuj_groq_odgovor(poruke):
    response_stream = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=poruke,
        temperature=0.2,
        max_tokens=1024,
        stream=True
    )
    
    for chunk in response_stream:
        if chunk.choices and chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content

# ----------------- BOČNI MENI (SIDEBAR) -----------------
with st.sidebar:
    st.image("https://pub-49fb3cc788a74e0a9edbac7e11305b94.r2.dev/biro_logo.jpg", use_container_width=True)
    st.title("🌲 Biro Chat")
    st.markdown("**Digitalni asistent Biroa za planiranje**\n\n*PD Srbijašume*")
    st.divider()
    
    st.markdown("### 🛠️ Status sistema")
    st.caption("🟢 **Vektorska baza:** Qdrant Cloud")
    st.caption("🟢 **LLM:** Groq Llama-3.3-70b")
    st.caption("🟢 **Embeddings:** MiniLM-L12-v2")
    st.caption(f"{'🟢' if HAS_RERANKER else '🟡'} **Reranker:** {'Aktivan' if HAS_RERANKER else 'Fallback heuristika'}")
    
    st.divider()
    
    if st.button("🔄 Osveži keš baze", use_container_width=True):
        st.cache_data.clear()
        st.success("Keš je osvežen!")

    if st.button("🧹 Obriši razgovor", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ----------------- GLAVNO ZAGLAVLJE -----------------
st.markdown("""
<div class="main-header">
    <h1>🌲 Biro za planiranje</h1>
    <p>PD Srbijašume — Digitalni asistent</p>
</div>
""", unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = []

# ----------------- TRAJNA BRZA PITANJA (EXPANDER) -----------------
with st.expander("💡 Brza predložena pitanja (kliknite da postavite)", expanded=(len(st.session_state.messages) == 0)):
    col1, col2, col3, col4 = st.columns(4)
    clicked_prompt = None
    if col1.button("👔 Ko je direktor?", use_container_width=True):
        clicked_prompt = "Ko je direktor Biroa i pokaži njegovu sliku?"
    if col2.button("👥 Ko su zamenici?", use_container_width=True):
        clicked_prompt = "Ko su zamenici direktora u Birou i prikaži njihove slike?"
    if col3.button("🌲 Crni vrh?", use_container_width=True):
        clicked_prompt = "Postoji li Crni vrh u bazi i šta piše o njemu?"
    if col4.button("📜 Član 114. Kol. ugovora?", use_container_width=True):
        clicked_prompt = "Navedi član 114. kolektivnog ugovora?"

    if clicked_prompt:
        st.session_state.prompt_input = clicked_prompt

# ----------------- PRIKAZ ISTORIJE PORUKA -----------------
for msg in st.session_state.messages:
    avatar = "👤" if msg["role"] == "user" else "🌲"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])
        if "image_url" in msg and msg["image_url"]:
            for img in msg["image_url"]:
                st.image(img, width=300)

# ----------------- OBRADA UNOSA KORISNIKA -----------------
prompt = st.chat_input("Postavite pitanje...")

if "prompt_input" in st.session_state and st.session_state.prompt_input:
    prompt = st.session_state.prompt_input
    del st.session_state.prompt_input

if prompt:
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar="🌲"):
        with st.spinner("Pretražujem bazu i generišem odgovor..."):
            try:
                kontekst, br_kandidata, ukupno_keširano, slike_urls = dobij_hibridni_kontekst(prompt)

                system_instruction = (
                    "Ti si stručni digitalni asistent Biroa za planiranje (PD Srbijašume).\n"
                    "Odgovaraj na pitanja korisnika isključivo na osnovu dostavljenog konteksta iz baze podataka.\n"
                    "Piši isključivo ispravnom srpskom latinicom (Gajevicom).\n"
                    "Budi koristan, precizan i jasan. Koristi podnaslove (`###`) i uređene liste gde god je to prikladno.\n"
                    "Ukoliko podatak zaista ne postoji u datom kontekstu, slobodno to naglasi korisniku sopstvenim rečima, ali uvek daj maksimum informacija koje jesu pronađene."
                )

                poruke_za_groq = [{"role": "system", "content": system_instruction}]
                
                skracena_istorija = st.session_state.messages[-4:]
                for msg in skracena_istorija:
                    poruke_za_groq.append({"role": msg["role"], "content": msg["content"]})
                
                upit_sa_kontekstom = f"KONTEKST IZ BAZE:\n{kontekst}\n\nTrenutno korisničko pitanje: {prompt}"
                poruke_za_groq.append({"role": "user", "content": upit_sa_kontekstom})
                
                odgovor = st.write_stream(strimuj_groq_odgovor(poruke_za_groq))
                
                validne_slike_za_prikaz = slike_urls[:2]
                for url in validne_slike_za_prikaz:
                     st.image(url, width=300, caption="Pronađena referenca u bazi")

                with st.expander("🔍 Pregled metapodataka pretrage"):
                    st.caption(f"Ukupno odlomaka u kešu: **{ukupno_keširano}**")
                    st.caption(f"Razmotreno rangiranih kandidata: **{br_kandidata}**")
                    st.caption(f"Korišćeni AI Model: **Groq Llama-3.3-70b**")
                    if validne_slike_za_prikaz:
                         st.caption(f"Pronađene vizuelne reference (URL): {', '.join(validne_slike_za_prikaz)}")
                    st.text_area("Pročišćen tekstualni kontekst iz baze:", value=kontekst, height=220)

                st.session_state.messages.append({"role": "user", "content": prompt})
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": odgovor,
                    "image_url": validne_slike_za_prikaz 
                })

            except Exception as e:
                st.error(f"Došlo je do greške u komunikaciji sa Groq-om ili bazom: {e}")