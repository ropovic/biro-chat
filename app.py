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

# ----------------- POPRAVLJENA UNIVERZALNA NORMALIZACIJA TEKSTA -----------------
def sredi_tekst(tekst):
    if not tekst:
        return ""
    tekst = str(tekst)
    tekst = tekst.replace('Љ', 'Lj').replace('љ', 'lj').replace('Њ', 'Nj').replace('њ', 'nj').replace('Џ', 'Dž').replace('џ', 'dž')
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
def pronadji_tacne_clanove(svi_odlomci, brojevi, upit_low):
    rezultati = []
    svi_pogodjeni_idx = set()
    trazi_kolektivni = any(w in upit_low for w in ["kolektivn", "ugovor", "ku", "kol."])
    
    for br in brojevi:
        br_str = str(br).strip()
        # Hvata "član 14", "član 18", "čl. 14", "člana 14"
        pattern_clan = r'\b(?:član|clan|članu|clanu|člana|clana|čl|cl)[a-z]*\.?\s*' + re.escape(br_str) + r'\b'
        
        for idx, item in enumerate(svi_odlomci):
            txt_low = item["tekst"].lower()
            izv_low = item["izvor"].lower()
            
            ima_eksplicitno = re.search(pattern_clan, txt_low)
            izvor_ku = any(w in izv_low for w in ["kolektivn", "ugovor", "ku"])
            ima_broj = re.search(r'\b' + re.escape(br_str) + r'\b', txt_low)
            
            if ima_eksplicitno or (trazi_kolektivni and izvor_ku and ima_broj):
                if idx not in svi_pogodjeni_idx:
                    svi_pogodjeni_idx.add(idx)
                    
                    spojeni_txt = ""
                    start_idx = max(0, idx - 1)
                    end_idx = min(len(svi_odlomci), idx + 3)
                    
                    for i in range(start_idx, end_idx):
                        if svi_odlomci[i]["izvor"] == item["izvor"]:
                            spojeni_txt += svi_odlomci[i]["tekst"] + "\n\n"
                        
                    rezultati.append({
                        "tekst": spojeni_txt.strip(),
                        "izvor": item["izvor"],
                        "slika_url": item.get("slika_url", ""),
                        "je_trazeni_clan": True
                    })
    return rezultati

# ----------------- NAPREDNA KONTROLA SLIKA ZA SVE ZAPOSLENE -----------------
def dobij_slike_za_upit(upit_low, svi_odlomci, izabrani_kandidati):
    imena_i_prezimena = [
        "nenad", "brana", "vamović", "vamovic", "biljana", "mirković", "mirkovic",
        "aleksandra", "katić", "katic", "arsenije", "simić", "simic", "bojana", "jelić", "jelic",
        "boško", "bosko", "malešević", "malesevic", "darko", "živanović", "zivanovic",
        "dragana", "miladinović", "miladinovic", "svetlana", "mihajlović", "mihajlovic",
        "goran", "ćaldović", "caldovic"
    ]
    
    je_zamenik = "zamenik" in upit_low or "zamenici" in upit_low
    je_direktor = ("direktor" in upit_low or "rukovodilac" in upit_low) and not je_zamenik
    pominje_osobu = any(w in upit_low for w in imena_i_prezimena + ["zaposlen", "zaposleni", "koga", "ko je"])
    trazi_sliku = any(w in upit_low for w in ["slika", "slike", "sliku", "foto", "fotografij", "izgleda", "pokaži", "prikazi"])
    
    pronadjene_slike = []
    vidjeni_url = set()
    
    def dodaj_sliku(url, caption):
        if url and url.startswith("http") and url not in vidjeni_url:
            pronadjene_slike.append((url, caption))
            vidjeni_url.add(url)

    # 1. Proveri kandidat odlomke koji su izabrani
    for item in izabrani_kandidati:
        url = item[3]
        txt_l = item[1].lower()
        if url and url.startswith("http"):
            dodaj_sliku(url, f"Fotografija iz odlomka. Detalji: {item[1][:300]}")

    # 2. Skener za pretragu po svim keširanim odlomcima ako tražimo konkretnu osobu
    if pominje_osobu or je_direktor or je_zamenik or trazi_sliku:
        pogodjena_imena = [w for w in imena_i_prezimena if w in upit_low]
        
        for item in svi_odlomci:
            txt_l = item["tekst"].lower()
            izv_l = item["izvor"].lower()
            url = item.get("slika_url", "").strip()
            
            if url and url.startswith("http"):
                if pogodjena_imena and any(ime in txt_l or ime in izv_l for ime in pogodjena_imena):
                    dodaj_sliku(url, f"Zvanična fotografija u bazi. Detalji: {item['tekst'][:300]}")
                elif je_direktor and ("direktor" in txt_l or "rukovodilac" in txt_l):
                    dodaj_sliku(url, f"Zvanična fotografija direktora. Detalji: {item['tekst'][:300]}")
                elif je_zamenik and ("zamenik" in txt_l or "zamenici" in txt_l):
                    dodaj_sliku(url, f"Zvanična fotografija zamenika. Detalji: {item['tekst'][:300]}")

    return pronadjene_slike[:4]

# ----------------- FILTRIRANJE I RANGIRANJE KANDIDATA -----------------
def filtriraj_i_skoruj_kandidate(svi_kandidati, upit):
    upit_low = sredi_tekst(upit).lower()
    
    je_direktor = ("direktor" in upit_low or "rukovodilac" in upit_low) and not ("zamenik" in upit_low or "zamenici" in upit_low)
    je_zamenik = "zamenik" in upit_low or "zamenici" in upit_low
    trazi_kolektivni = any(w in upit_low for w in ["kolektivn", "ugovor", "ugovora", "kol."])
    
    crni_oblici = ["crni", "crnog", "crnom", "crnim", "crne"]
    vrh_oblici = ["vrh", "vrha", "vrhu", "vrhom", "vrhovi", "vrhova"]
    trazi_crni_vrh = any(w in upit_low for w in crni_oblici) and any(w in upit_low for w in vrh_oblici)
    
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

        if item.get("je_trazeni_clan") or item.get("je_lokacija") or item.get("je_osoba_iz_upita"):
            skor += 5000000

        if je_direktor:
            if any(w in txt_low or w in izvor_low for w in ["direktor", "rukovodilac", "rukovodenje", "biro"]):
                skor += 3000000
            if any(w in txt_low or w in izvor_low for w in ["gj", "vranjača", "vranjaca", "osnova gazdovanja"]):
                skor -= 2000000  

        if je_zamenik:
            if any(w in txt_low or w in izvor_low for w in ["zamenik", "svetlana", "goran", "ćaldović", "caldovic", "mihajlović"]):
                skor += 3000000

        if trazi_crni_vrh:
            ima_crni = any(c in txt_low or c in izvor_low for c in crni_oblici)
            ima_vrh = any(v in txt_low or v in izvor_low for v in vrh_oblici)
            if ima_crni and ima_vrh:
                skor += 5000000

        if brojevi:
            for br in brojevi:
                pattern_clan = r'\b(?:član|clan|članu|clanu|člana|clana|čl|cl)[a-z]*\.?\s*' + re.escape(str(br)) + r'\b'
                if re.search(pattern_clan, txt_low):
                    skor += 5000000
                elif re.search(r'\b' + re.escape(str(br)) + r'\b', txt_low):
                    skor += 1000000

        if trazi_kolektivni:
            if any(w in izvor_low for w in ["kolektiv", "ku", "ugovor"]):
                skor += 1000000

        if vazne_reci:
            br_pogodaka = sum(1 for rec in vazne_reci if rec in txt_low or rec in izvor_low)
            skor += br_pogodaka * 20000

        skorovani_kandidati.append((skor, txt, izvor, slika_url))

    skorovani_kandidati.sort(key=lambda x: x[0], reverse=True)
    return skorovani_kandidati[:12]

# ----------------- HIBRIDNA PRETRAGA SA FORCE-INJECTION -----------------
def dobij_hibridni_kontekst(upit, top_k_rezultata=10, max_karaktera=7000):
    svi_odlomci = ucitaj_sve_tekstove()
    svi_kandidati = []
    svi_vidjeni = set()
    norm_upit = sredi_tekst(upit)
    upit_low = norm_upit.lower()
    
    brojevi = re.findall(r'\b\d+\b', upit)
    je_pretraga_clana = bool(brojevi and any(w in upit_low for w in ["clan", "član", "cl", "čl", "ugovor", "kolektivni"]))
    
    je_zamenik = "zamenik" in upit_low or "zamenici" in upit_low
    je_direktor = ("direktor" in upit_low or "rukovodilac" in upit_low) and not je_zamenik
    
    crni_oblici = ["crni", "crnog", "crnom", "crnim", "crne"]
    vrh_oblici = ["vrh", "vrha", "vrhu", "vrhom", "vrhovi", "vrhova"]
    trazi_crni_vrh = any(w in upit_low for w in crni_oblici) and any(w in upit_low for w in vrh_oblici)
    
    imena_i_prezimena = [
        "nenad", "brana", "vamović", "vamovic", "biljana", "mirković", "mirkovic",
        "aleksandra", "katić", "katic", "arsenije", "simić", "simic", "bojana", "jelić", "jelic",
        "boško", "bosko", "malešević", "malesevic", "darko", "živanović", "zivanovic",
        "dragana", "miladinović", "miladinovic", "svetlana", "mihajlović", "mihajlovic",
        "goran", "ćaldović", "caldovic"
    ]
    pominje_osobu = any(w in upit_low for w in imena_i_prezimena)

    # 1. FORCE-INJECTION (Garantovano ubacivanje na vrh)
    if je_pretraga_clana or brojevi:
        direktni_clanovi = pronadji_tacne_clanove(svi_odlomci, brojevi, upit_low)
        for dc in direktni_clanovi:
            if dc["tekst"] not in svi_vidjeni:
                svi_vidjeni.add(dc["tekst"])
                svi_kandidati.insert(0, dc)

    if pominje_osobu or je_direktor or je_zamenik:
        for item in svi_odlomci:
            txt_l = item["tekst"].lower()
            izv_l = item["izvor"].lower()
            if any(name in upit_low and (name in txt_l or name in izv_l) for name in imena_i_prezimena):
                if item["tekst"] not in svi_vidjeni:
                    svi_vidjeni.add(item["tekst"])
                    svi_kandidati.insert(0, {
                        "tekst": item["tekst"],
                        "izvor": item["izvor"],
                        "slika_url": item.get("slika_url", ""),
                        "je_osoba_iz_upita": True
                    })

    if trazi_crni_vrh:
        for item in svi_odlomci:
            txt_l = item["tekst"].lower()
            izv_l = item["izvor"].lower()
            ima_crni = any(c in txt_l or c in izv_l for c in crni_oblici)
            ima_vrh = any(v in txt_l or v in izv_l for v in vrh_oblici)
            if ima_crni and ima_vrh:
                if item["tekst"] not in svi_vidjeni:
                    svi_vidjeni.add(item["tekst"])
                    svi_kandidati.insert(0, {
                        "tekst": item["tekst"],
                        "izvor": item["izvor"],
                        "slika_url": item.get("slika_url", ""),
                        "je_lokacija": True
                    })

    # 2. VEKTORSKA PRETRAGA
    try:
        query_vector = list(embed_model.embed([norm_upit]))[0].tolist()
        points = qdrant.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_vector,
            limit=25
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
                            "slika_url": str(slika_url).strip()
                        })
    except Exception:
        pass

    if not svi_kandidati and svi_odlomci:
        svi_kandidati = svi_odlomci[:10]

    skorovani = filtriraj_i_skoruj_kandidate(svi_kandidati, upit)
    izabrani = skorovani[:top_k_rezultata]

    slike_podaci = dobij_slike_za_upit(upit_low, svi_odlomci, izabrani)

    kontekst_lista = []
    
    for url, cap in slike_podaci:
        kontekst_lista.append(f"Zvanična vizuelna referenca u bazi za ovaj upit:\n{cap}")

    for skor, txt, izvor, slika_url in izabrani:
        cist_txt = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', txt)
        chunk_str = f"Odlomak iz dokumenta [{izvor}]:\n{cist_txt.strip()}"
        kontekst_lista.append(chunk_str)

    spojeni_tekst = "\n\n---\n\n".join(kontekst_lista)

    if len(spojeni_tekst) > max_karaktera:
        spojeni_tekst = spojeni_tekst[:max_karaktera] + "\n...[Skraćeno]"

    return spojeni_tekst, len(skorovani), len(svi_odlomci), slike_podaci

# ----------------- STRIMOVANJE GROQ ODGOVORA -----------------
def strimuj_groq_odgovor(poruke):
    try:
        response_stream = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=poruke,
            temperature=0.1,
            max_tokens=900,
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
                max_tokens=900,
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
    if col4.button("📜 Članovi 14 i 18?", use_container_width=True):
        clicked_prompt = "Navedi član 14 i član 18 Kolektivnog ugovora."

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
                    "Odgovaraj na pitanja ISKLJUČIVO na osnovu dostavljenog KONTEKSTA.\n\n"
                    "STRIKTNA PRAVILA PROTIV HALUCINACIJA (IZMIŠLJANJA):\n"
                    "1. NIKADA nemoj izmišljati sadržaj članova ugovora, zakona ili podatke o zaposlenima i lokacijama!\n"
                    "2. Ako se traženi član (npr. član 14 ili član 18) ili informacija NE NALAZI u dostavljenom kontekstu, IZRIČITO i pošteno napiši da taj član/podatak ne postoji u bazi ili nije dostavljen u kontekstu. NEMOJ sastavljati tekst ugovora iz glave!\n"
                    "3. Budi izuzetno precizan oko lokacija (npr. Crni vrh): nemoj tvrditi da lokacija pripada nekoj gazdinskoj jedinici (npr. Vranjača - Dijelovi) osim ako to EKSPLICITNO ne piše u tekstu odlomka.\n"
                    "4. ZABRANJENO JE ispisivati URL linkove (http...) ili formatirati slike preko Markdown koda (![slika](...)). Aplikacija će sama prikazati fotografiju ispod teksta.\n"
                    "Piši isključivo srpskim jezikom i latinicom."
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