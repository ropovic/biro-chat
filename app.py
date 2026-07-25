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
    
    reranker_model = None
    if HAS_RERANKER:
        try:
            reranker_model = TextRerank(model_name="BAAI/bge-reranker-base")
        except Exception:
            reranker_model = None
            
    return qdrant, groq, embed_model, reranker_model

qdrant, groq, embed_model, reranker_model = init_clients()

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
                
                if raw_txt:
                    sve_tacke.append({
                        "tekst": sredi_tekst(raw_txt),
                        "izvor": sredi_tekst(izvor)
                    })
        
        if next_offset is None or len(records) == 0:
            break
            
        offset = next_offset

    return sve_tacke

# ----------------- HELPERI ZA ČLANOVE I SADRŽAJ -----------------
def pretvori_u_rimske(broj_str):
    mapping = {
        "1": "I", "2": "II", "3": "III", "4": "IV", "5": "V",
        "6": "VI", "7": "VII", "8": "VIII", "9": "IX", "10": "X",
        "11": "XI", "12": "XII", "13": "XIII", "14": "XIV", "15": "XV"
    }
    return mapping.get(broj_str, "")

def je_pogodak_za_clan(txt_low, broj_str):
    """Proverava prisustvo reči član/čl uz traženi broj."""
    rimski = pretvori_u_rimske(broj_str)
    pats = [
        rf'\b(član|clan|čl|cl)\.?\s*0?{broj_str}\b',
        rf'\b0?{broj_str}\b\.\s*(član|clan)\b'
    ]
    if rimski:
        pats.append(rf'\b(član|clan|čl|cl)\.?\s*{rimski}\b')
        
    for pat in pats:
        if re.search(pat, txt_low):
            return True
    return False

def je_sadrzaj_toc(txt_low):
    if "sadržaj" in txt_low or "sadrzaj" in txt_low:
        return True
    matches = re.findall(r'\.\.\.\s*\d+|\b\d+\s*$', txt_low, re.MULTILINE)
    if len(matches) >= 3:
        return True
    return False

# ----------------- FILTRIRANJE I BODOVANJE KANDIDATA -----------------
def filtriraj_i_skoruj_kandidate(svi_kandidati, upit):
    upit_low = upit.lower()
    brojevi = re.findall(r'\b\d+\b', upit)
    
    je_kolektivni = any(w in upit_low for w in ["kolektivn", "ugovor"])
    je_clan = any(w in upit_low for w in ["clan", "član", "cl", "čl"])
    je_direktor = any(w in upit_low for w in ["direktor", "zamenik", "zamenici", "rukovodstv", "uprava", "sef", "šef"])

    skorovani_kandidati = []

    for item in svi_kandidati:
        txt = item["tekst"]
        izvor = item["izvor"]
        txt_low = txt.lower()
        izvor_low = izvor.lower()
        skor = 0

        if (je_kolektivni or je_direktor or je_clan) and "<table>" in txt_low and "kolektivn" not in txt_low and "direktor" not in txt_low:
            continue

        if (je_kolektivni or je_clan) and je_sadrzaj_toc(txt_low):
            skor -= 5000

        if je_kolektivni or je_clan:
            if "kolektivn" in izvor_low or "kolektivn" in txt_low:
                skor += 300

            if brojevi and je_clan:
                for br in brojevi:
                    if je_pogodak_za_clan(txt_low, br):
                        if "kolektivn" in izvor_low or "kolektivn" in txt_low:
                            skor += 10000
                        else:
                            skor += 2000
                    elif f"{br}." in txt_low and "kolektivn" in izvor_low:
                        skor += 1500

        if je_direktor:
            if "zamenik" in txt_low or "direktor" in txt_low:
                skor += 1000
            if "http" in txt_low:
                skor += 2000

        upit_reci = [r for r in upit_low.split() if len(r) > 3]
        for rec in upit_reci:
            if rec in txt_low:
                skor += 10

        skorovani_kandidati.append((skor, txt, izvor))

    skorovani_kandidati.sort(key=lambda x: x[0], reverse=True)
    return skorovani_kandidati[:15]

# ----------------- HIBRIDNA PRETRAGA SA SPAJANJEM SUSREDNIH ČANAKA -----------------
def dobij_hibridni_kontekst(upit, top_k_rezultata=6, max_karaktera=4000):
    svi_odlomci = ucitaj_sve_tekstove()
    svi_kandidati = []
    svi_vidjeni = set()
    norm_upit = sredi_tekst(upit)
    brojevi = re.findall(r'\b\d+\b', upit)
    upit_low = norm_upit.lower()

    # Ako tražimo član, automatski spajamo i sledeće odlomke radi kompletnosti pojmova
    if brojevi and any(w in upit_low for w in ["clan", "član", "cl", "čl"]):
        for br in brojevi:
            for idx, item in enumerate(svi_odlomci):
                txt = item["tekst"]
                izvor = item["izvor"]
                txt_low = txt.lower()
                izvor_low = izvor.lower()

                if je_sadrzaj_toc(txt_low):
                    continue

                if je_pogodak_za_clan(txt_low, br) or (f"{br}." in txt_low and "kolektivn" in izvor_low):
                    if ("kolektivn" in upit_low and ("kolektivn" in izvor_low or "kolektivn" in txt_low)) or "kolektivn" not in upit_low:
                        # Spajamo trenutni i naredna 2 odlomka iz ugovora da definicije ne bi ostale odsečene
                        prosirani_tekst = txt
                        for step in range(1, 3):
                            if idx + step < len(svi_odlomci):
                                sledeci_item = svi_odlomci[idx + step]
                                if "kolektivn" in sledeci_item["izvor"].lower() or "kolektivn" in izvor_low:
                                    prosirani_tekst += "\n" + sledeci_item["tekst"]

                        if prosirani_tekst not in svi_vidjeni:
                            svi_kandidati.append({"tekst": prosirani_tekst, "izvor": izvor})
                            svi_vidjeni.add(prosirani_tekst)

    query_vector = list(embed_model.embed([norm_upit]))[0].tolist()
    vector_response = qdrant.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=25
    )
    for hit in vector_response.points:
        if hit.payload:
            raw_txt = (hit.payload.get("tekst") or hit.payload.get("text") or 
                       hit.payload.get("content") or hit.payload.get("page_content") or "")
            izvor = (hit.payload.get("naziv_dokumenta") or hit.payload.get("file_name") or 
                     hit.payload.get("izvor") or hit.payload.get("dokument") or 
                     hit.payload.get("source") or "")
            if raw_txt:
                norm_txt = sredi_tekst(raw_txt)
                if norm_txt not in svi_vidjeni:
                    svi_vidjeni.add(norm_txt)
                    svi_kandidati.append({
                        "tekst": norm_txt,
                        "izvor": sredi_tekst(izvor)
                    })

    if not svi_kandidati:
        return "", 0, len(svi_odlomci)

    skorovani = filtriraj_i_skoruj_kandidate(svi_kandidati, upit)

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
        if txt.startswith("Izvor"):
            kontekst_lista.append(txt)
        else:
            kontekst_lista.append(f"Odlomak iz baze:\n{txt}")

    spojeni_tekst = "\n\n--- ODLOMAK IZ BAZE ---\n\n".join(kontekst_lista)

    if len(spojeni_tekst) > max_karaktera:
        spojeni_tekst = spojeni_tekst[:max_karaktera] + "\n...[Kontekst skraćen radi limita]..."

    return spojeni_tekst, len(skorovani), len(svi_odlomci)

# ----------------- BOČNI MENI (SIDEBAR) -----------------
with st.sidebar:
    st.image("https://pub-49fb3cc788a74e0a9edbac7e11305b94.r2.dev/biro_logo.jpg", use_container_width=True)
    st.title("🌲 Biro Chat")
    st.markdown("**Digitalni asistent Biroa za planiranje**\n\n*PD Srbijašume*")
    st.divider()
    
    st.markdown("### 🛠️ Status sistema")
    st.caption("🟢 **Vektorska baza:** Qdrant Cloud")
    st.caption("🟢 **LLM:** Llama-3.3 / Llama-3.1 (Auto-Fallback)")
    st.caption("🟢 **Embeddings:** FastEmbed")
    st.caption(f"{'🟢' if HAS_RERANKER else '🟡'} **Reranker:** {'Aktivan' if HAS_RERANKER else 'Fallback heurisitka'}")
    
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
        clicked_prompt = "Ko su zamenici direktora u Birou i prikaži njihove slike?"
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

# ----------------- OBRADA UNOSA KORISNIKA SA FALLBACK SVOJSTVOM -----------------
prompt = st.chat_input("Postavite pitanje...")

if "prompt_input" in st.session_state and st.session_state.prompt_input:
    prompt = st.session_state.prompt_input
    del st.session_state.prompt_input

if prompt:
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar="🌲"):
        with st.spinner("Pretražujem i rangiram podatke..."):
            try:
                kontekst, br_kandidata, ukupno_keširano = dobij_hibridni_kontekst(prompt)

                system_prompt = (
                    "Ti si asistent Biroa za planiranje (PD Srbijašume).\n"
                    "Odgovaraj tačno i direktno na osnovu konteksta iz baze.\n\n"
                    "VAŽNO PRAVILO ZA SLIKE:\n"
                    "Ako se u kontekstu nalazi bilo koji URL slike (npr. http...jpg ili .png), "
                    "MORAŠ ga prikazati u odgovoru kao Markdown sliku: ![Opis slike](URL).\n\n"
                    "OSTALA PRAVILA:\n"
                    "1. Odgovaraj na srpskom jeziku (latinica).\n"
                    "2. Ako traženi član ili podatak NE POSTOJI u kontekstu, napiši tačno: "
                    "'Traženi član/podatak se ne nalazi u dostupnim izvodima dokumenta u bazi.'\n"
                    "3. Koristi podnaslove (`###`) i liste sa boldovanim rečima.\n\n"
                    f"KONTEKST IZ BAZE:\n{kontekst}"
                )

                messages_for_groq = [{"role": "system", "content": system_prompt}]
                
                skracena_istorija = st.session_state.messages[-4:]
                for msg in skracena_istorija:
                    messages_for_groq.append({"role": msg["role"], "content": msg["content"]})
                
                messages_for_groq.append({"role": "user", "content": prompt})

                modeli = ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"]
                odgovor = None
                korisceni_model = ""

                for m in modeli:
                    try:
                        response = groq.chat.completions.create(
                            model=m,
                            messages=messages_for_groq,
                            temperature=0.1
                        )
                        odgovor = response.choices[0].message.content
                        korisceni_model = m
                        break
                    except Exception as err:
                        if "rate_limit_exceeded" in str(err).lower() or "429" in str(err):
                            continue
                        else:
                            raise err

                if odgovor is None:
                    odgovor = "Trenutno su svi AI modeli preopterećeni. Molimo vas pokušajte ponovo za nekoliko minuta."

                st.markdown(odgovor)

                # DEBUG EXPANDER
                with st.expander("🔍 Pregled pročišćenog konteksta poslatog Llami"):
                    st.caption(f"Ukupno odlomaka u kešu: **{ukupno_keširano}**")
                    st.caption(f"Razmotreno rangiranih kandidata: **{br_kandidata}**")
                    st.caption(f"Korišćeni AI Model: **{korisceni_model}**")
                    st.text_area("Sadržaj poslat Llami:", value=kontekst, height=220)

                st.session_state.messages.append({"role": "user", "content": prompt})
                st.session_state.messages.append({"role": "assistant", "content": odgovor})

            except Exception as e:
                st.error(f"Došlo je do greške: {e}")