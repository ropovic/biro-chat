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
    try:
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
    except Exception as e:
        st.error(f"Greška prilikom inicijalizacije klijenata: {e}")
        st.stop()

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
        'Ћ': 'Ć', 'У': 'U', 'Ф': 'F', 'Х': 'H', 'Ц': 'C', 'Č': 'Č', 'Š': 'Š'
    }
    return "".join([zamene.get(ch, ch) for ch in tekst])

# ----------------- BRZO KEŠIRANJE SVIH ODLOMAKA IZ BAZE -----------------
@st.cache_data(ttl=1800)
def ucitaj_sve_tekstove():
    sve_tacke = []
    offset = None
    try:
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
                               r.payload.get("content") or r.payload.get("page_content") or 
                               r.payload.get("body") or "")
                    
                    izvor = (r.payload.get("naziv_dokumenta") or r.payload.get("file_name") or  
                             r.payload.get("izvor") or r.payload.get("dokument") or  
                             r.payload.get("source") or "")
                    
                    slika_url = (r.payload.get("slika_url") or r.payload.get("image_url") or 
                                 r.payload.get("slika") or r.payload.get("photo_url") or 
                                 r.payload.get("url") or "")
                    
                    if not slika_url and raw_txt:
                        img_match = re.search(r'(https?://[^\s<>"]+?\.(?:jpg|jpeg|png|webp|gif))', raw_txt, re.IGNORECASE)
                        if img_match:
                            slika_url = img_match.group(1)
                    
                    if raw_txt:
                        sve_tacke.append({
                            "tekst": sredi_tekst(raw_txt),
                            "izvor": sredi_tekst(izvor),
                            "slika_url": str(slika_url).strip()
                        })
            
            if next_offset is None or len(records) == 0:
                break
                
            offset = next_offset
    except Exception as e:
        st.warning(f"Upozorenje pri učitavanju baze: {e}")

    return sve_tacke

# ----------------- UNAPREĐENO DETEKTOVANJE ČLANOVA -----------------
def pronadji_tacne_clanove(svi_odlomci, brojevi):
    rezultati = []
    svi_pogodjeni_idx = set()
    
    for br in brojevi:
        br_str = str(br).strip()
        p1 = r'\b(član[uaem]?|clan[uaem]?|čl\.?|cl\.?)\s*(?:br\.?)?\s*' + re.escape(br_str) + r'\b'
        p2 = r'\b' + re.escape(br_str) + r'\.\s*(član[uaem]?|clan[uaem]?)\b'
        
        for idx, item in enumerate(svi_odlomci):
            txt_low = item["tekst"].lower()
            if re.search(p1, txt_low) or re.search(p2, txt_low) or (br_str in txt_low and ("clan" in txt_low or "član" in txt_low)):
                if idx not in svi_pogodjeni_idx:
                    svi_pogodjeni_idx.add(idx)
                    spojeni_txt = item["tekst"]
                    if idx + 1 < len(svi_odlomci):
                        spojeni_txt += "\n" + svi_odlomci[idx + 1]["tekst"]
                        
                    rezultati.append({
                        "tekst": spojeni_txt,
                        "izvor": item["izvor"],
                        "slika_url": item.get("slika_url", ""),
                        "je_trazeni_clan": True
                    })
    return rezultati

# ----------------- PAMETNA KONTROLA SLIKA -----------------
def dobij_slike_za_upit(upit_low, svi_odlomci, izabrani_kandidati):
    je_zamenik = "zamenik" in upit_low or "zamenici" in upit_low
    je_direktor = ("direktor" in upit_low or "brano" in upit_low or "vamović" in upit_low or "vamovic" in upit_low) and not je_zamenik
    trazi_sliku = any(w in upit_low for w in ["slika", "slike", "sliku", "foto", "fotografij", "izgleda", "pokaži", "prikazi"])
    
    if not (je_zamenik or je_direktor or trazi_sliku):
        return []

    pronadjene_slike = []
    vidjeni_url = set()
    
    def dodaj_sliku(url, caption):
        if url and url.startswith("http") and url not in vidjeni_url:
            pronadjene_slike.append((url, caption))
            vidjeni_url.add(url)
            
    if je_zamenik:
        imamo_sv = False
        imamo_go = False
        for item in svi_odlomci:
            txt_l = item["tekst"].lower()
            url = item.get("slika_url", "").strip()
            if url and url.startswith("http"):
                if not imamo_sv and any(w in txt_l for w in ["svetlana", "mihajlović", "mihajlovic"]):
                    dodaj_sliku(url, "Zvanična fotografija zamenika — Svetlana Mihajlović")
                    imamo_sv = True
                elif not imamo_go and any(w in txt_l for w in ["goran", "ćaldović", "caldovic"]):
                    dodaj_sliku(url, "Zvanična fotografija zamenika — Goran Ćaldović")
                    imamo_go = True
            if imamo_sv and imamo_go: break
            
    elif je_direktor:
        for item in svi_odlomci:
            txt_l = item["tekst"].lower()
            url = item.get("slika_url", "").strip()
            if url and url.startswith("http"):
                if any(w in txt_l for w in ["brano", "vamović", "vamovic", "direktor"]):
                    dodaj_sliku(url, "Zvanična fotografija direktora — Brano Vamović")
                    break
    else:
        for item in izabrani_kandidati:
            url = item[3]
            dodaj_sliku(url, "Povezana fotografija iz baze")
            if len(pronadjene_slike) >= 2: break
            
    return pronadjene_slike[:2]

# ----------------- FILTRIRANJE I RANGIRANJE KANDIDATA -----------------
def filtriraj_i_skoruj_kandidate(svi_kandidati, upit):
    upit_low = sredi_tekst(upit).lower()
    trazi_kolektivni = any(w in upit_low for w in ["kolektivn", "ugovor", "ugovora", "kol."])
    je_zamenik = "zamenik" in upit_low or "zamenici" in upit_low
    je_direktor = ("direktor" in upit_low or "brano" in upit_low or "vamović" in upit_low or "vamovic" in upit_low) and not je_zamenik
    
    # Provera za Crni vrh ili druge specifične lokacije
    trazi_crni_vrh = "crni" in upit_low and "vrh" in upit_low
    brojevi = re.findall(r'\b\d+\b', upit)

    STOP_WORDS = {
        "ko", "je", "su", "prikazi", "prikazuj", "pokaži", "sliku", "slika", "foto", 
        "o", "u", "i", "da", "li", "clan", "član", "kolektivnog", "kolektivni", "ugovora", 
        "ugovor", "zamenik", "zamenici", "direktor", "ima", "bazi", "postoji", "šta", 
        "sta", "piše", "pise", "njemu", "njoj", "imama", "na", "sa", "za", "od", "do", 
        "iz", "se", "ne", "bi", "gde", "kad", "kako", "zašto", "zasto", "koji", "koja", 
        "koje", "koju", "kojim", "kojih", "radi", "daje", "daju", "navedi", "daj", "nadji",
        "pronadji", "nađi", "traži", "trazi", "opis", "opisi", "podatak", "podataka", "bazu"
    }
    
    vazne_reci = [w for w in re.findall(r'\b\w+\b', upit_low) if len(w) > 2 and w not in STOP_WORDS and not w.isdigit()]

    skorovani_kandidati = []

    for item in svi_kandidati:
        txt = item["tekst"]
        izvor = item["izvor"]
        slika_url = item.get("slika_url", "")
        txt_low = txt.lower()
        izvor_low = izvor.lower()
        skor = 10 

        # SUPREMI PRORITET ZA TRAŽENE ČLANOVE (Npr. Član 114)
        if item.get("je_trazeni_clan"):
            skor += 500000

        if item.get("je_osoba_iz_upita") or item.get("je_tacan_pogodak_fraze"):
            skor += 100000

        # SUPREMI PRIORITET ZA LOKACIJE (Npr. Crni vrh)
        if trazi_crni_vrh:
            if "crni" in txt_low and "vrh" in txt_low:
                skor += 400000
            if "crni" in izvor_low or "vrh" in izvor_low:
                skor += 150000

        if brojevi:
            for br in brojevi:
                if br in txt_low:
                    skor += 50000
                if br in izvor_low:
                    skor += 20000

        if trazi_kolektivni:
            if "kolektiv" in izvor_low or "ku" in izvor_low or "ugovor" in izvor_low:
                skor += 80000
            if "kolektiv" in txt_low or "ugovor" in txt_low:
                skor += 30000
            if "pdv" in txt_low or "pdv" in izvor_low or "porez" in txt_low:
                skor -= 200000 

        if vazne_reci:
            br_pogodaka = sum(1 for rec in vazne_reci if rec in txt_low or rec in izvor_low)
            if br_pogodaka == len(vazne_reci):
                skor += 60000 
            else:
                skor += br_pogodaka * 3000

        if je_zamenik:
            if any(w in txt_low or w in izvor_low for w in ["zamenik", "svetlana", "goran", "ćaldović", "caldovic", "mihajlović"]):
                skor += 60000
            if "brano" in txt_low or "vamović" in txt_low:
                skor -= 20000

        if je_direktor:
            if "direktor" in txt_low or "brano" in txt_low or "vamović" in txt_low:
                skor += 60000

        skorovani_kandidati.append((skor, txt, izvor, slika_url))

    skorovani_kandidati.sort(key=lambda x: x[0], reverse=True)
    return skorovani_kandidati[:12]

# ----------------- HIBRIDNA PRETRAGA -----------------
def dobij_hibridni_kontekst(upit, top_k_rezultata=8, max_karaktera=4000):
    svi_odlomci = ucitaj_sve_tekstove()
    svi_kandidati = []
    svi_vidjeni = set()
    norm_upit = sredi_tekst(upit)
    upit_low = norm_upit.lower()
    
    brojevi = re.findall(r'\b\d+\b', upit)
    je_pretraga_clana = bool(brojevi and any(w in upit_low for w in ["clan", "član", "cl", "čl"]))
    
    je_zamenik = "zamenik" in upit_low or "zamenici" in upit_low
    je_direktor = ("direktor" in upit_low or "brano" in upit_low or "vamović" in upit_low or "vamovic" in upit_low) and not je_zamenik
    zamenici_keywords = ["svetlana", "goran", "caldovic", "ćaldović", "mihajlović", "mihajlovic"]

    STOP_WORDS = {
        "ko", "je", "su", "prikazi", "prikazuj", "pokaži", "sliku", "slika", "foto", 
        "o", "u", "i", "da", "li", "clan", "član", "kolektivnog", "kolektivni", "ugovora", 
        "ugovor", "zamenik", "zamenici", "direktor", "ima", "bazi", "postoji", "šta", 
        "sta", "piše", "pise", "njemu", "njoj", "imama", "na", "sa", "za", "od", "do", 
        "iz", "se", "ne", "bi", "gde", "kad", "kako", "zašto", "zasto", "koji", "koja", 
        "koje", "koju", "kojim", "kojih", "radi", "daje", "daju", "navedi", "daj", "nadji",
        "pronadji", "nađi", "traži", "trazi", "opis", "opisi", "podatak", "podataka", "bazu"
    }
    
    vazne_reci = [w for w in re.findall(r'\b\w+\b', upit_low) if len(w) > 2 and w not in STOP_WORDS and not w.isdigit()]

    if je_pretraga_clana:
        direktni_pogodci = pronadji_tacne_clanove(svi_odlomci, brojevi)
        for dp in direktni_pogodci:
            if dp["tekst"] not in svi_vidjeni:
                svi_vidjeni.add(dp["tekst"])
                svi_kandidati.insert(0, dp)

    if svi_odlomci:
        for item in svi_odlomci:
            txt_l = item["tekst"].lower()
            izv_l = item["izvor"].lower()

            if je_zamenik:
                if any(k in txt_l or k in izv_l for k in zamenici_keywords + ["zamenik"]):
                    if item["tekst"] not in svi_vidjeni:
                        svi_vidjeni.add(item["tekst"])
                        kand = item.copy()
                        kand["je_osoba_iz_upita"] = True
                        svi_kandidati.insert(0, kand)

            elif je_direktor:
                if any(k in txt_l or k in izv_l for k in ["brano", "vamović", "vamovic"]):
                    if item["tekst"] not in svi_vidjeni:
                        svi_vidjeni.add(item["tekst"])
                        kand = item.copy()
                        kand["je_osoba_iz_upita"] = True
                        svi_kandidati.insert(0, kand)

            elif vazne_reci and all(rec in txt_l or rec in izv_l for rec in vazne_reci):
                if item["tekst"] not in svi_vidjeni:
                    svi_vidjeni.add(item["tekst"])
                    kand = item.copy()
                    kand["je_tacan_pogodak_fraze"] = True
                    svi_kandidati.insert(0, kand)

    try:
        query_vector = list(embed_model.embed([norm_upit]))[0].tolist()
        points = qdrant.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_vector,
            limit=20
        )

        for hit in points:
            if hit.payload:
                raw_txt = (hit.payload.get("tekst") or hit.payload.get("text") or hit.payload.get("content") or "")
                izvor = (hit.payload.get("naziv_dokumenta") or hit.payload.get("file_name") or hit.payload.get("izvor") or "")
                slika_url = (hit.payload.get("slika_url") or hit.payload.get("image_url") or hit.payload.get("slika") or "")
                
                if raw_txt:
                    norm_txt = sredi_tekst(raw_txt)
                    if norm_txt not in svi_vidjeni:
                        svi_vidjeni.add(norm_txt)
                        svi_kandidati.append({
                            "tekst": norm_txt,
                            "izvor": sredi_tekst(izvor),
                            "slika_url": str(slika_url).strip(),
                            "je_trazeni_clan": False,
                            "je_osoba_iz_upita": False,
                            "je_tacan_pogodak_fraze": False
                        })
    except Exception:
        pass

    if not svi_kandidati and svi_odlomci:
        svi_kandidati = svi_odlomci[:10]

    skorovani = filtriraj_i_skoruj_kandidate(svi_kandidati, upit)
    izabrani = skorovani[:top_k_rezultata]

    kontekst_lista = []
    for skor, txt, izvor, slika_url in izabrani:
        cist_txt = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', txt)
        chunk_str = f"Odlomak (Izvor: {izvor}):\n{cist_txt.strip()}"
        kontekst_lista.append(chunk_str)

    spojeni_tekst = "\n\n---\n\n".join(kontekst_lista)

    if len(spojeni_tekst) > max_karaktera:
        spojeni_tekst = spojeni_tekst[:max_karaktera] + "\n...[Skraćeno]"

    slike_podaci = dobij_slike_za_upit(upit_low, svi_odlomci, izabrani)

    return spojeni_tekst, len(skorovani), len(svi_odlomci), slike_podaci

# ----------------- STRIMOVANJE GROQ ODGOVORA -----------------
def strimuj_groq_odgovor(poruke):
    try:
        response_stream = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=poruke,
            temperature=0.1,
            max_tokens=800,
            stream=True
        )
        for chunk in response_stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
    except Exception as e:
        if "429" in str(e) or "rate_limit" in str(e).lower():
            st.toast("⚠️ Dostignut limit. Prebačeno na 8B!", icon="🔄")
            response_stream = groq_client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=poruke,
                temperature=0.1,
                max_tokens=800,
                stream=True
            )
            for chunk in response_stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        else:
            raise e

# ----------------- BOČNI MENI -----------------
with st.sidebar:
    st.image("https://pub-49fb3cc788a74e0a9edbac7e11305b94.r2.dev/biro_logo.jpg", use_container_width=True)
    st.title("🌲 Biro Chat")
    st.markdown("**Digitalni asistent Biroa za planiranje**\n\n*PD Srbijašume*")
    st.divider()
    
    st.markdown("### 🛠️ Status sistema")
    st.caption("🟢 **Vektorska baza:** Qdrant Cloud")
    st.caption("🟢 **LLM:** Groq Llama (Auto-fallback 70B ➔ 8B)")
    st.caption("🟢 **Embeddings:** MiniLM-L12-v2")
    
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
        
        if "image_data" in msg and msg["image_data"]:
            for url, cap in msg["image_data"]:
                st.image(url, width=300, caption=cap)
        elif "image_url" in msg and msg["image_url"]:
            for item in msg["image_url"]:
                if isinstance(item, tuple):
                    st.image(item[0], width=300, caption=item[1])
                else:
                    st.image(item, width=300)

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
                kontekst, br_kandidata, ukupno_keširano, slike_podaci = dobij_hibridni_kontekst(prompt)

                system_instruction = (
                    "Ti si stručni digitalni asistent Biroa za planiranje (PD Srbijašume).\n"
                    "Odgovaraj na pitanja ISKLJUČIVO na osnovu dostavljenog KONTEKSTA.\n"
                    "STRIKTNA PRAVILA:\n"
                    "1. ZABRANJENO JE ispisivati URL linkove (http...) ili formatirati slike preko Markdown koda (![slika](...)). Aplikacija će sama prikazati fotografiju ispod teksta.\n"
                    "2. Ako nađeš traženi podatak (npr. član ugovora, lokaciju, Crni vrh), jasno i precizno ga citiraj i objasni na osnovu teksta u bazi.\n"
                    "3. Ako korisnik pita za Kolektivni ugovor i član se nalazi u kontekstu, tačno ga navedi bez ubacivanja nebitnih zakona.\n"
                    "Piši isključivo srpskom latinicom."
                )

                poruke_za_groq = [{"role": "system", "content": system_instruction}]
                
                skracena_istorija = st.session_state.messages[-2:]
                for msg in skracena_istorija:
                    poruke_za_groq.append({"role": msg["role"], "content": msg["content"]})
                
                upit_sa_kontekstom = f"KONTEKST IZ BAZE:\n{kontekst}\n\nTrenutno korisničko pitanje: {prompt}"
                poruke_za_groq.append({"role": "user", "content": upit_sa_kontekstom})
                
                odgovor = st.write_stream(strimuj_groq_odgovor(poruke_za_groq))
                
                for url, cap in slike_podaci:
                    st.image(url, width=300, caption=cap)

                with st.expander("🔍 Pregled metapodataka pretrage"):
                    st.caption(f"Ukupno odlomaka u kešu: **{ukupno_keširano}**")
                    st.caption(f"Razmotreno rangiranih kandidata: **{br_kandidata}**")
                    if slike_podaci:
                         st.caption(f"Prikazana vizuelna referenca: {len(slike_podaci)}")
                    st.text_area("Pročišćen tekstualni kontekst iz baze:", value=kontekst, height=200)

                st.session_state.messages.append({"role": "user", "content": prompt})
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": odgovor,
                    "image_data": slike_podaci 
                })

            except Exception as e:
                st.error(f"Došlo je do greške u komunikaciji: {e}")