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
    
    res = [zamene.get(ch, ch) for ch in tekst]
    return "".join(res)

# ----------------- DEDUPLIKACIJA SLIKA PO OSOBI -----------------
def izvuci_jedinstvene_slike_po_osobi(lista_slika):
    odabrane_slike = []
    vidjene_osobe = set()
    
    for url in lista_slika:
        if not url or not isinstance(url, str):
            continue
        url_clean = url.strip()
        if not url_clean.startswith("http"):
            continue
            
        url_low = url_clean.lower()
        
        # Identifikujemo tačnu osobu da ne dupliramo slike iste osobe
        if "svetlana" in url_low:
            osoba_key = "svetlana"
        elif "goran" in url_low or "cald" in url_low:
            osoba_key = "goran"
        elif "brano" in url_low or "vano" in url_low:
            osoba_key = "brano"
        else:
            osoba_key = url_low.split("/")[-1].split("?")[0]
            
        if osoba_key not in vidjene_osobe:
            vidjene_osobe.add(osoba_key)
            odabrane_slike.append(url_clean)
            
    return odabrane_slike

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
def pronadji_tacne_clanove(svi_odlomci, brojevi, trazi_kolektivni=False):
    rezultati = []
    svi_pogodjeni_idx = set()
    
    for br in brojevi:
        br_str = str(br).strip()
        p1 = r'\b(član[uaem]?|clan[uaem]?|čl\.?|cl\.?)\s*(?:br\.?)?\s*' + re.escape(br_str) + r'\b'
        p2 = r'\b' + re.escape(br_str) + r'\.\s*(član[uaem]?|clan[uaem]?)\b'
        
        for idx, item in enumerate(svi_odlomci):
            txt_low = item["tekst"].lower()
            izv_low = item["izvor"].lower()
            
            if re.search(p1, txt_low) or re.search(p2, txt_low):
                # Ako korisnik traži Kolektivni ugovor, ignorišemo odlomke koji pripadaju drugim zakonima (npr. PDV)
                if trazi_kolektivni and not ("kolektivn" in txt_low or "ugovor" in txt_low or "kolektivn" in izv_low):
                    continue
                    
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

# ----------------- FILTRIRANJE I RANGIRANJE KANDIDATA -----------------
def filtriraj_i_skoruj_kandidate(svi_kandidati, upit, trazi_kolektivni=False):
    upit_low = upit.lower()
    
    je_zamenik = "zamenik" in upit_low or "zamenici" in upit_low
    je_direktor = ("direktor" in upit_low or "brano" in upit_low or "vamović" in upit_low or "vamovic" in upit_low) and not je_zamenik

    STOP_WORDS = {
        "ko", "je", "su", "prikazi", "prikazuj", "pokaži", "sliku", "slika", "foto", 
        "o", "u", "i", "da", "li", "clan", "član", "kolektivnog", "kolektivni", "ugovora", 
        "ugovor", "zamenik", "zamenici", "direktor", "ima", "bazi", "postoji", "šta", 
        "sta", "piše", "pise", "njemu", "njoj", "imama", "na", "sa", "za", "od", "do", 
        "iz", "se", "ne", "bi", "gde", "kad", "kako", "zašto", "zasto", "koji", "koja", 
        "koje", "koju", "kojim", "kojih", "radi", "daje", "daju"
    }
    vazne_reci = [w for w in re.findall(r'\b\w+\b', sredi_tekst(upit_low)) if len(w) > 2 and w not in STOP_WORDS]

    skorovani_kandidati = []

    for item in svi_kandidati:
        txt = item["tekst"]
        izvor = item["izvor"]
        slika_url = item.get("slika_url", "")
        txt_low = txt.lower()
        izvor_low = izvor.lower()
        skor = 10 

        if item.get("je_trazeni_clan"):
            skor += 100000

        if item.get("je_osoba_iz_upita"):
            skor += 50000

        # Ako korisnik pita o Kolektivnom ugovoru, dajemo prednost Kolektivnom ugovoru i skidam bodove PDV-u
        if trazi_kolektivni:
            if "kolektivn" in txt_low or "ugovor" in txt_low or "kolektivn" in izvor_low:
                skor += 20000
            elif "pdv" in txt_low or "zakon" in txt_low:
                skor -= 50000 

        # Provera ključnih fraza (npr. "crni" i "vrh")
        if vazne_reci:
            br_pogodaka = sum(1 for rec in vazne_reci if rec in txt_low or rec in izvor_low)
            if br_pogodaka == len(vazne_reci):
                skor += 40000 # Ako su SVE reči prisutne u odlomku
            else:
                skor += br_pogodaka * 1000

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
    return skorovani_kandidati[:10]

# ----------------- HIBRIDNA PRETRAGA SA FLEKSIBILNIM UČITAVANJEM -----------------
def dobij_hibridni_kontekst(upit, top_k_rezultata=5, max_karaktera=3500):
    svi_odlomci = ucitaj_sve_tekstove()
    svi_kandidati = []
    svi_vidjeni = set()
    norm_upit = sredi_tekst(upit)
    upit_low = norm_upit.lower()
    
    brojevi = re.findall(r'\b\d+\b', upit)
    je_pretraga_clana = bool(brojevi and any(w in upit_low for w in ["clan", "član", "cl", "čl"]))
    trazi_kolektivni = any(w in upit_low for w in ["kolektivn", "ugovor", "ugovora", "kol."])
    
    je_zamenik = "zamenik" in upit_low or "zamenici" in upit_low
    je_direktor = ("direktor" in upit_low or "brano" in upit_low or "vamović" in upit_low or "vamovic" in upit_low) and not je_zamenik

    pronadjene_slike_urls = []

    zamenici_keywords = ["svetlana", "goran", "caldovic", "ćaldović", "mihajlović", "mihajlovic"]

    STOP_WORDS = {
        "ko", "je", "su", "prikazi", "prikazuj", "pokaži", "sliku", "slika", "foto", 
        "o", "u", "i", "da", "li", "clan", "član", "kolektivnog", "kolektivni", "ugovora", 
        "ugovor", "zamenik", "zamenici", "direktor", "ima", "bazi", "postoji", "šta", 
        "sta", "piše", "pise", "njemu", "njoj", "imama", "na", "sa", "za", "od", "do", 
        "iz", "se", "ne", "bi", "gde", "kad", "kako", "zašto", "zasto", "koji", "koja", 
        "koje", "koju", "kojim", "kojih", "radi", "daje", "daju"
    }
    vazne_reci = [w for w in re.findall(r'\b\w+\b', norm_upit) if len(w) > 2 and w not in STOP_WORDS]

    if svi_odlomci:
        for item in svi_odlomci:
            txt_l = item["tekst"].lower()
            izv_l = item["izvor"].lower()
            s_url = item.get("slika_url", "")

            # Pretrazivanje po osobi / zameniku
            if je_zamenik:
                if any(k in txt_l or k in izv_l or k in s_url.lower() for k in zamenici_keywords + ["zamenik"]):
                    if item["tekst"] not in svi_vidjeni:
                        svi_vidjeni.add(item["tekst"])
                        kand = item.copy()
                        kand["je_osoba_iz_upita"] = True
                        svi_kandidati.insert(0, kand)
                        if s_url and any(k in s_url.lower() for k in zamenici_keywords):
                            pronadjene_slike_urls.append(s_url)

            elif je_direktor:
                if any(k in txt_l or k in izv_l or k in s_url.lower() for k in ["brano", "vamović", "vamovic"]):
                    if item["tekst"] not in svi_vidjeni:
                        svi_vidjeni.add(item["tekst"])
                        kand = item.copy()
                        kand["je_osoba_iz_upita"] = True
                        svi_kandidati.insert(0, kand)
                        if s_url and "brano" in s_url.lower():
                            pronadjene_slike_urls.append(s_url)

            # Traženje po ključnim pojmovima (npr. Crni vrh)
            elif vazne_reci and all(rec in txt_l or rec in izv_l for rec in vazne_reci):
                if item["tekst"] not in svi_vidjeni:
                    svi_vidjeni.add(item["tekst"])
                    kand = item.copy()
                    kand["je_osoba_iz_upita"] = True
                    svi_kandidati.insert(0, kand)
                    if s_url:
                        pronadjene_slike_urls.append(s_url)

    # Pretraga članova u keširanoj bazi
    if je_pretraga_clana:
        direktni_pogodci = pronadji_tacne_clanove(svi_odlomci, brojevi, trazi_kolektivni=trazi_kolektivni)
        for dp in direktni_pogodci:
            if dp["tekst"] not in svi_vidjeni:
                svi_vidjeni.add(dp["tekst"])
                svi_kandidati.insert(0, dp)

    # Vektorska dopuna pretrage preko Qdrant-a
    try:
        query_vector = list(embed_model.embed([norm_upit]))[0].tolist()
        points = qdrant.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_vector,
            limit=15
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
                            "je_osoba_iz_upita": False
                        })
    except Exception:
        pass

    if not svi_kandidati and svi_odlomci:
        svi_kandidati = svi_odlomci[:10]

    skorovani = filtriraj_i_skoruj_kandidate(svi_kandidati, upit, trazi_kolektivni=trazi_kolektivni)
    izabrani = skorovani[:top_k_rezultata]

    kontekst_lista = []
    for skor, txt, izvor, slika_url in izabrani:
        chunk_str = f"Odlomak (Izvor: {izvor}):\n{txt}"
        kontekst_lista.append(chunk_str)

    spojeni_tekst = "\n\n---\n\n".join(kontekst_lista)

    if len(spojeni_tekst) > max_karaktera:
        spojeni_tekst = spojeni_tekst[:max_karaktera] + "\n...[Skraćeno]"

    čiste_slike = izvuci_jedinstvene_slike_po_osobi(pronadjene_slike_urls)
    return spojeni_tekst, len(skorovani), len(svi_odlomci), čiste_slike

# ----------------- STRIMOVANJE GROQ ODGOVORA SA AUTOMATSKIM FALLBACK-OM -----------------
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
            st.toast("⚠️ Dostignut limit za 70B model. Prebačeno na Llama-3.1-8B!", icon="🔄")
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

# ----------------- TRAJNA BRZA PITANJA -----------------
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
                    "STRIKTNA PRAVILA ZA ODGOVARANJE:\n"
                    "1. Ako korisnik pita za određeni dokument (npr. Kolektivni ugovor), odgovaraj ISKLJUČIVO o tom dokumentu. Ako traženi član tog dokumenta NE POSTOJI u dostavljenom kontekstu, odgovori kratko: 'Član [broj] Kolektivnog ugovora nije pronađen u bazi podataka.' NIKADA nemoj spominjati druge nebitne zakone ili odlomke (poput Zakona o PDV) ako korisnik pita o Kolektivnom ugovoru!\n"
                    "2. Ako se traženi član NALAZI u kontekstu, detaljno i tačno ga citiraj i objasni.\n"
                    "3. NIKADA nemoj izmišljati povezanost između različitih članova niti spominjati članove koji nisu traženi.\n"
                    "4. VAŽNO O SLIKAMA: Nikada nemoj tvrditi da si tekstualni asistent ili da ne možeš prikazati slike. Aplikacija automatski prikazuje fotografije ispod tvog odgovora ako postoje u bazi.\n"
                    "Budi koristan, precizan i lak za čitanje. Koristi podnaslove (`###`) i liste gde je to potrebno."
                )

                poruke_za_groq = [{"role": "system", "content": system_instruction}]
                
                skracena_istorija = st.session_state.messages[-2:]
                for msg in skracena_istorija:
                    poruke_za_groq.append({"role": msg["role"], "content": msg["content"]})
                
                upit_sa_kontekstom = f"KONTEKST IZ BAZE:\n{kontekst}\n\nTrenutno korisničko pitanje: {prompt}"
                poruke_za_groq.append({"role": "user", "content": upit_sa_kontekstom})
                
                odgovor = st.write_stream(strimuj_groq_odgovor(poruke_za_groq))
                
                validne_slike_za_prikaz = izvuci_jedinstvene_slike_po_osobi(slike_urls)[:2] 
                for url in validne_slike_za_prikaz:
                     caption_text = "Zvanična fotografija zaposlenog"
                     url_low = url.lower()
                     if "vano" in url_low or "brano" in url_low:
                         caption_text = "Zvanična fotografija direktora — Brano Vamović"
                     elif "svetlana" in url_low:
                         caption_text = "Zvanična fotografija zamenika — Svetlana Mihajlović"
                     elif "goran" in url_low or "cald" in url_low:
                         caption_text = "Zvanična fotografija zamenika — Goran Ćaldović"
                     st.image(url, width=300, caption=caption_text)

                with st.expander("🔍 Pregled metapodataka pretrage"):
                    st.caption(f"Ukupno odlomaka u kešu: **{ukupno_keširano}**")
                    st.caption(f"Razmotreno rangiranih kandidata: **{br_kandidata}**")
                    if validne_slike_za_prikaz:
                         st.caption(f"Pronađena vizuelna referenca (URL): {', '.join(validne_slike_za_prikaz)}")
                    st.text_area("Pročišćen tekstualni kontekst iz baze:", value=kontekst, height=200)

                st.session_state.messages.append({"role": "user", "content": prompt})
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": odgovor,
                    "image_url": validne_slike_za_prikaz 
                })

            except Exception as e:
                st.error(f"Došlo je do greške u komunikaciji: {e}")