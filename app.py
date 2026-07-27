import streamlit as st
import re
from qdrant_client import QdrantClient
from groq import Groq
from fastembed import TextEmbedding

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
        return qdrant, groq_client, embed_model
    except Exception as e:
        st.error(f"Greška prilikom inicijalizacije klijenata: {e}")
        st.stop()

qdrant, groq_client, embed_model = init_clients()

# ----------------- NORMALIZACIJA TEKSTA -----------------
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

def ukloni_dijakritike(tekst):
    if not tekst:
        return ""
    zamene = {
        'č': 'c', 'ć': 'c', 'š': 's', 'ž': 'z', 'đ': 'd',
        'Č': 'c', 'Ć': 'c', 'Š': 's', 'Ž': 'z', 'Đ': 'd'
    }
    txt = sredi_tekst(tekst).lower()
    return "".join([zamene.get(ch, ch) for ch in txt])

# ----------------- KEŠIRANJE SVIH ODLOMAKA IZ BAZE -----------------
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
                        norm_txt = sredi_tekst(raw_txt)
                        norm_izv = sredi_tekst(izvor)
                        sve_tacke.append({
                            "tekst": norm_txt,
                            "tekst_ascii": ukloni_dijakritike(norm_txt),
                            "izvor": norm_izv,
                            "izvor_ascii": ukloni_dijakritike(norm_izv),
                            "slika_url": str(slika_url).strip()
                        })
            
            if next_offset is None or len(records) == 0:
                break
                
            offset = next_offset
    except Exception as e:
        st.warning(f"Upozorenje pri učitavanju baze: {e}")

    return sve_tacke

# ----------------- STROGI FILTER ZA SLIKE DIREKTORA/ZAMENIKA -----------------
def filtriraj_slike_za_prikaz(upit_ascii, rangirani_kandidati):
    prikazi_slike = []
    vidjene = set()

    je_zamenik = "zamenik" in upit_ascii or "zamenici" in upit_ascii
    je_direktor = ("direktor" in upit_ascii or "rukovodilac" in upit_ascii) and not je_zamenik

    for entry in rangirani_kandidati:
        item = entry["item"]
        url = item.get("slika_url", "").strip()
        txt_a = item["tekst_ascii"]
        izv_a = item["izvor_ascii"]

        if not url or not url.startswith("http") or url in vidjene:
            continue

        if je_direktor:
            # Slike zamenika su striktno zabranjene ako tražimo samo direktora
            if "zamenik" not in txt_a and any(k in txt_a or k in izv_a for k in ["direktor", "darko", "zivanovic"]):
                prikazi_slike.append((url, f"Fotografija direktora. Izvor: {item['izvor']}"))
                vidjene.add(url)
        elif je_zamenik:
            if any(k in txt_a or k in izv_a for k in ["zamenik", "zamenici", "svetlana", "goran", "mihajlovic", "caldovic"]):
                prikazi_slike.append((url, f"Fotografija zamenika direktora. Izvor: {item['izvor']}"))
                vidjene.add(url)
        else:
            imena_mapa = ["nenad", "veres", "biljana", "mirkovic", "brana", "vamovic", "aleksandra", "katic"]
            if any(ime in upit_ascii and ime in txt_a for ime in imena_mapa):
                prikazi_slike.append((url, f"Fotografija. Izvor: {item['izvor']}"))
                vidjene.add(url)

    return prikazi_slike[:3]

# ----------------- DETERMINISTIČKI HIBRIDNI PRETRAŽIVAČ -----------------
def dobij_hibridni_kontekst(upit, top_k_rezultata=8, max_karaktera=7000):
    svi_odlomci = ucitaj_sve_tekstove()
    upit_ascii = ukloni_dijakritike(upit)
    norm_upit = sredi_tekst(upit)
    
    candidates_map = {}

    # 1. DETEKCIJA ČLANOVA (Član 14, Član 18)
    brojevi = re.findall(r'\b\d+\b', upit)
    je_clan_upit = any(w in upit_ascii for w in ["clan", "cl", "ugovor", "kolektivni", "ku"])
    
    if brojevi and je_clan_upit:
        for br in brojevi:
            pattern = r'\b(?:clan|cl)[a-z]*\.?\s*' + re.escape(str(br)) + r'\b'
            for item in svi_odlomci:
                if re.search(pattern, item["tekst_ascii"]):
                    key = item["tekst"]
                    if key not in candidates_map:
                        candidates_map[key] = {"item": item, "score": 200000.0}
                    else:
                        candidates_map[key]["score"] += 100000.0

    # 2. PROXIMITY REGEX ZA FRAZU "Crni vrh" (Crni vrh, Crnog vrha, Crnom vrhu)
    if "crn" in upit_ascii and "vrh" in upit_ascii:
        crni_vrh_pattern = r'\bcrn[a-z]*\s+(?:[a-z]+\s+)?vrh[a-z]*\b'
        for item in svi_odlomci:
            if re.search(crni_vrh_pattern, item["tekst_ascii"]):
                key = item["tekst"]
                if key not in candidates_map:
                    candidates_map[key] = {"item": item, "score": 200000.0}
                else:
                    candidates_map[key]["score"] += 100000.0

    # 3. DETEKCIJA ULOGA (Direktor vs Zamenik)
    je_zamenik = "zamenik" in upit_ascii or "zamenici" in upit_ascii
    je_direktor = ("direktor" in upit_ascii or "rukovodilac" in upit_ascii) and not je_zamenik

    if je_direktor:
        for item in svi_odlomci:
            txt_a = item["tekst_ascii"]
            if "zamenik" not in txt_a and any(w in txt_a for w in ["direktor", "darko", "zivanovic"]):
                key = item["tekst"]
                if key not in candidates_map:
                    candidates_map[key] = {"item": item, "score": 150000.0}
                else:
                    candidates_map[key]["score"] += 50000.0

    elif je_zamenik:
        for item in svi_odlomci:
            txt_a = item["tekst_ascii"]
            if any(w in txt_a for w in ["zamenik", "zamenici", "svetlana", "mihajlovic", "goran", "caldovic"]):
                key = item["tekst"]
                if key not in candidates_map:
                    candidates_map[key] = {"item": item, "score": 150000.0}
                else:
                    candidates_map[key]["score"] += 50000.0

    # 4. VEKTORSKA PRETRAGA (QDRANT)
    try:
        query_vector = list(embed_model.embed([norm_upit]))[0].tolist()
        points = qdrant.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_vector,
            limit=25
        )
        for rank, hit in enumerate(points):
            if hit.payload:
                raw_txt = (hit.payload.get("tekst") or hit.payload.get("text") or hit.payload.get("content") or "")
                izvor = (hit.payload.get("naziv_dokumenta") or hit.payload.get("file_name") or hit.payload.get("izvor") or "")
                slika_url = (hit.payload.get("slika_url") or hit.payload.get("image_url") or hit.payload.get("slika") or "")
                
                if raw_txt:
                    norm_txt = sredi_tekst(raw_txt)
                    vec_score = (25 - rank) * 100.0
                    
                    if norm_txt in candidates_map:
                        candidates_map[norm_txt]["score"] += vec_score
                    else:
                        candidates_map[norm_txt] = {
                            "item": {
                                "tekst": norm_txt,
                                "tekst_ascii": ukloni_dijakritike(norm_txt),
                                "izvor": sredi_tekst(izvor),
                                "izvor_ascii": ukloni_dijakritike(izvor),
                                "slika_url": str(slika_url).strip()
                            },
                            "score": vec_score
                        }
    except Exception:
        pass

    if not candidates_map and svi_odlomci:
        for item in svi_odlomci[:10]:
            candidates_map[item["tekst"]] = {"item": item, "score": 10.0}

    # 5. DODATNI KEYWORD MATCHING
    stop_reci = {"ko", "je", "su", "sta", "pise", "bazi", "postoji", "navedi", "prikazi", "pokazi", "u", "i", "na", "sa", "za", "o"}
    reci_upita = [w for w in re.findall(r'\b\w+\b', upit_ascii) if len(w) > 2 and w not in stop_reci]
    
    for key, data in candidates_map.items():
        txt_a = data["item"]["tekst_ascii"]
        izv_a = data["item"]["izvor_ascii"]
        match_count = sum(1 for w in reci_upita if w in txt_a or w in izv_a)
        data["score"] += match_count * 150.0

    # RANGIRANJE
    rangirani = sorted(candidates_map.values(), key=lambda x: x["score"], reverse=True)
    top_k = [entry["item"] for entry in rangirani[:top_k_rezultata]]

    # DOBIJANJE TAČNIH SLIKA
    slike_za_prikaz = filtriraj_slike_za_prikaz(upit_ascii, rangirani)

    # FORMIRANJE KONTEKSTA ZA LLM
    kontekst_delovi = []
    for item in top_k:
        cist_txt = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', item["tekst"])
        kontekst_delovi.append(f"Odlomak iz dokumenta [{item['izvor']}]:\n{cist_txt.strip()}")

    spojeni_tekst = "\n\n---\n\n".join(kontekst_delovi)
    if len(spojeni_tekst) > max_karaktera:
        spojeni_tekst = spojeni_tekst[:max_karaktera] + "\n...[Skraćeno]"

    return spojeni_tekst, len(rangirani), len(svi_odlomci), slike_za_prikaz

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
    st.caption("🟢 **Reranker:** Hybrid Proximity Engine v4")
    
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
                    "Odgovaraj na pitanja na osnovu dostavljenog KONTEKSTA.\n\n"
                    "SMERNICE ZA ODGOVARANJE:\n"
                    "1. Kada korisnik traži članove ugovora (npr. Član 14 ili Član 18) ili informacije o lokaciji (npr. Crni vrh), precizno i detaljno prenesi ono što piše u dostavljenim odlomcima.\n"
                    "2. Ako se podatak ili član zaista NE NALAZI u dostavljenom kontekstu, pošteno saopšti da podatak nije dostupan u bazi. NEMOJ izmišljati teksta članova!\n"
                    "3. STROGO JE ZABRANJENO ispisivanje URL linkova za slike u tekstu odgovora ili formatiranje slika preko Markdown koda (![slika](...)). Aplikacija će sama prikazati fotografiju ispod odgovora.\n"
                    "Odgovaraj isključivo na srpskom jeziku, latinicom."
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
                    st.caption(f"Rangiranih kandidata: **{br_kandidata}**")
                    if slike_podaci:
                         st.caption(f"Prikazana vizuelna referenca: {len(slike_podaci)}")
                    st.text_area("Pročišćen tekstualni kontekst poslat modelu:", value=kontekst, height=200)

                st.session_state.messages.append({"role": "user", "content": prompt})
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": odgovor,
                    "image_data": slike_podaci 
                })

            except Exception as e:
                st.error(f"Došlo je do greške u komunikaciji: {e}")