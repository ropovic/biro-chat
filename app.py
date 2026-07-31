import streamlit as st
import re
import time
from qdrant_client import QdrantClient, models
from groq import Groq
from fastembed import TextEmbedding

# ============================================================
# KONFIGURACIJA STRANICE
# ============================================================
st.set_page_config(
    page_title="Биро Чат Асистент",
    page_icon="🌲",
    layout="centered"
)

# ============================================================
# EMBEDDING KONFIGURACIJA — STAGE 2
# ============================================================
# Stari model (Stage 1):  sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2  (384 dim)
# Novi model (Stage 2):  sentence-transformers/paraphrase-multilingual-mpnet-base-v2  (768 dim)
#
# VAŽNO: Stara kolekcija "baza_cloud_v2" ima 384-dim vektore. Ona je NEKOMPATIBILNA
# sa novim modelom. Pokreni reindex.py jednom da napravi "baza_cloud_v2_e5" sa 768-dim
# vektorima, pa postavi COLLECTION_NAME = "baza_cloud_v2_e5" u Streamlit secrets.
#
# Alternativa (bolji kvalitet, ali ~2x RAM):
#   EMBEDDING_MODEL = "intfloat/multilingual-e5-large"   # 1024 dim, 2.24 GB
#   Za e5-large treba dodati "query: " / "passage: " prefikse u embed_upit() i reindex.py
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
EMBEDDING_DIM = 768

# ============================================================
# CUSTOM CSS DIZAJN
# ============================================================
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

# ============================================================
# UČITAVANJE KLJUČEVA
# ============================================================
potrebne_tajne = ["QDRANT_URL", "QDRANT_API_KEY", "GROQ_API_KEY"]
for tajna in potrebne_tajne:
    if tajna not in st.secrets:
        st.error(f"❌ Nedostaje ključ '{tajna}' u Streamlit Secrets-u!")
        st.stop()

QDRANT_URL = st.secrets["QDRANT_URL"]
QDRANT_API_KEY = st.secrets["QDRANT_API_KEY"]
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]

# Ime kolekcije se može override-ovati preko secrets; default je nova e5 kolekcija
COLLECTION_NAME = st.secrets.get("COLLECTION_NAME", "baza_cloud_v2_e5")

# ============================================================
# UNAPRED KOMPAJLIRANI REGEX IZRAZI I STOP REČI
# ============================================================
RE_URL = re.compile(r'(https?://[^\s<>"]+?\.(?:jpg|jpeg|png|webp|gif))', re.IGNORECASE)
RE_CLEAN_URL = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F])+)')

STOP_RECI = {
    "ko", "je", "su", "sta", "pise", "bazi", "postoji", "navedi", "prikazi", "pokazi",
    "slika", "slike", "sliku", "foto", "fotografij", "u", "i", "na", "sa", "za", "o",
    "da", "li", "ima", "njegovu", "njihove", "njena", "mesto", "radno", "biro", "biroa",
    "planiranje", "projektovanje", "pd", "srbijasume", "sumarstvu", "detalje", "detaljnije",
    "koji", "koja", "koje", "kog", "kojoj", "kojim", "svi", "sve", "svih", "kao", "ali",
    "ili", "gde", "kada", "kako", "ovaj", "ova", "ovo", "taj", "ta", "to", "vec", "samo",
    "jos", "vrlo", "neki", "neka", "neko", "nesto"
}

# ============================================================
# MAPIRANJE KLJUČNIH REČI -> KATEGORIJA DOKUMENTA
# ------------------------------------------------------------
# Ovo je ključno za rešavanje problema "ne nalazi štampače/tonere".
# Kad upit sadrži reč iz neke kategorije, automatski se primenjuje
# Qdrant filter na polje "tip" u payload-u.
#
# VAŽNO: Ove "tip" vrednosti moraju postojati u payload-u za filter
# da radi. Trenutno u bazi postoje "fotografija_profil" i "dijagram".
# Za ostalo, pokreni reindex.py koji automatski inferiše "tip" iz
# naziva dokumenta i sadržaja.
# ============================================================
KATEGORIJA_MAPPING = {
    "oprema": {
        "keywords": [
            "stampač", "stampac", "stampaci", "printer", "pisač", "pisac",
            "toner", "toneri", "kertridž", "kertridz", "kartuša", "cartridge",
            "skener", "monitor", "računar", "racunar", "kompjuter", "laptop",
            "miš", "mis", "tastatura", "oprema", "inventar", "mreža", "mreza",
            "štampa", "stampa", "server", "hardver", "uredjaj", "uređaj",
        ],
        "tip_values": [
            "oprema", "kancelarijska_oprema", "inventar", "toner", "stampac",
        ],
    },
    "kadrovski": {
        "keywords": [
            "direktor", "rukovodilac", "zaposleni", "radnik", "šef", "sef",
            "osoblje", "lice", "lica", "kadrovi", "imenik", "biografija",
            "fotografija", "profil",
        ],
        "tip_values": [
            "kadrovski", "zaposleni", "osoblje", "fotografija_profil", "biografija",
        ],
    },
    "pravni": {
        "keywords": [
            "član", "clan", "ugovor", "kolektivni", "pravilnik", "zakon",
            "propis", "odluka", "rešenje", "resenje", "statut", "članovi",
        ],
        "tip_values": [
            "pravni_akt", "ugovor", "kolektivni_ugovor", "pravilnik", "zakon", "odluka",
        ],
    },
    "projektna": {
        "keywords": [
            "gazdinska", "g.j.", "gj ", "osnova", "projekat", "šuma", "sume",
            "šumski", "sumski", "drvna", "drvno", "seča", "seca", "gazdinstv",
            "gospodarska", "jedinica",
        ],
        "tip_values": [
            "projektna_dokumentacija", "sumskoprivredna_osnova",
            "gospodarska_jedinica", "mapa", "karta",
        ],
    },
    "vizuel": {
        "keywords": [
            "dijagram", "grafikon", "šema", "shema", "mapa", "karta",
            "ruža vetrova", "ruza vetrova", "vetrova", "ruža",
            "crtež", "ilustracija", "skica", "prikaz", "tabela", "shema",
        ],
        "tip_values": [
            "dijagram", "mapa", "karta", "tabela", "grafikon", "vizuel",
        ],
    },
    "osoba": {
        "keywords": [
            "fotografija", "profil", "lica", "lice",
            "izgled", "portret", "slika zaposlenog", "slika osobe",
        ],
        "tip_values": ["fotografija_profil", "biografija"],
    },
}

# ============================================================
# INICIJALIZACIJA KLIJENATA
# ============================================================
@st.cache_resource
def init_clients():
    try:
        qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, check_compatibility=False)
        groq_client = Groq(api_key=GROQ_API_KEY)
        embed_model = TextEmbedding(model_name=EMBEDDING_MODEL)
        return qdrant, groq_client, embed_model
    except Exception as e:
        st.error(f"Greška prilikom inicijalizacije klijenata: {e}")
        st.stop()

qdrant, groq_client, embed_model = init_clients()

# ============================================================
# NORMALIZACIJA I KORENI (STEMOVANJE)
# ============================================================
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

def _stemuj_rec(w):
    if len(w) >= 7:
        return w[:-2]
    elif len(w) >= 5:
        return w[:-1]
    return w

STOP_KORENI = {_stemuj_rec(w) for w in STOP_RECI}

def izvuci_kljucne_reci(upit_ascii):
    return [w for w in re.findall(r'\b\w+\b', upit_ascii) if len(w) > 2]

def izvuci_korene(upit_ascii):
    reci = izvuci_kljucne_reci(upit_ascii)
    rezultat = []
    for w in reci:
        koren = _stemuj_rec(w)
        if koren in STOP_KORENI:
            continue
        skracen = len(koren) < len(w)
        rezultat.append((koren, skracen))
    return rezultat

def koren_prisutan(koren, skracen, tekst):
    if skracen:
        return re.search(r'\b' + re.escape(koren), tekst) is not None
    return re.search(r'\b' + re.escape(koren) + r'\b', tekst) is not None

# ============================================================
# EMBEDDING HELPER
# ------------------------------------------------------------
# Za mpnet model (default): nema prefiks
# Za e5-large (ako se uvede): treba "query: " prefiks ovde
#                              i "passage: " prefiks u reindex.py
# ============================================================
def embed_upit(tekst):
    """Embedding za korisnički upit. Za e5-large bi trebalo: f'query: {tekst}'."""
    if "e5" in EMBEDDING_MODEL.lower():
        tekst = f"query: {tekst}"
    return list(embed_model.embed([tekst]))[0].tolist()

# ============================================================
# DETEKCIJA KATEGORIJE IZ UPITA + QDRANT FILTER
# ============================================================
def je_eksplicitno_vizuelni_upit(upit):
    """Da li korisnik eksplicitno traži da mu se nešto vizuelno prikaže?"""
    upit_lower = (upit or "").lower()
    return any(r in upit_lower for r in [
        "prikaži", "prikazi", "pokaži", "pokazi",
        "daj mi sliku", "daj sliku", "prikaži sliku",
        "prikaži fotografiju", "prikaži dijagram",
        "prikaži šemu", "prikaži mapu", "prikaži crtež",
        "prikazati", "pokazati",
    ])


def detektuj_kategoriju(upit):
    """
    Na osnovu ključnih reči u upitu vraća listu vrednosti za 'tip' polje
    koje treba koristiti u Qdrant filteru. Vraća None ako nema detekcije.

    VAŽNO: "vizuel" kategorija se primenjuje SAMO kad korisnik EKSPLICITNO
    traži da mu se nešto prikaže ("prikaži", "pokaži"). Za informaciona pitanja
    tipa "Pominje li se X?" vizuel filter se NE primenjuje — jer tekst koji
    pominje X može biti u zapisu koji nema tip=dijagram, pa bi filter
    izbacio relevantan kontekst.
    """
    upit_lower = upit.lower()
    eksplicitno_vizuel = je_eksplicitno_vizuelni_upit(upit)

    matches = set()
    for kategorija, info in KATEGORIJA_MAPPING.items():
        for kw in info["keywords"]:
            if kw in upit_lower:
                # "vizuel" filter važi SAMO uz eksplicitnu nameru prikaza
                if kategorija == "vizuel" and not eksplicitno_vizuel:
                    break
                matches.update(info["tip_values"])
                break
    return list(matches) if matches else None

def napravi_qdrant_filter(tip_vrednosti):
    """Pravi Qdrant filter za 'tip' polje (OR logika)."""
    if not tip_vrednosti:
        return None
    return models.Filter(
        should=[
            models.FieldCondition(
                key="tip",
                match=models.MatchAny(any=tip_vrednosti),
            )
        ]
    )

# ============================================================
# KOJE TIPOVE SLIKA SMEMO DA PRIKAŽEMO ZA KOJI FILTER
# ------------------------------------------------------------
# Ovo sprečava da se uz tekstualni odgovor o nečemu (npr. "GJ Mrčajevac")
# prikažu fotografije zaposlenih koje su slučajno u top-K jer je dokument
# pomenuo neke osobe. Pravilo: slike se prikazuju SAMO kad su kontekstualno
# relevantne za dati filter/upit.
# ============================================================
KAT_SLIKA = {
    "vizuel":     {"dijagram", "mapa", "karta", "grafikon", "tabela"},
    "osoba":      {"fotografija_profil", "biografija"},
    "kadrovski":  {"fotografija_profil", "biografija"},
    "projektna":  {"mapa", "karta", "dijagram", "tabela"},
    "oprema":     set(),   # Oprema nema tipičan vizuelni sadržaj
    "pravni":     set(),   # Pravni akti nemaju vizuelni sadržaj
}

def dozvoljeni_tipovi_za_filter(aktivan_filter, eksplicitno_vizuel=False):
    """Za dati filter i eksplicitnost namere, vrati dozvoljene tipove slika."""
    if eksplicitno_vizuel:
        return {"fotografija_profil", "biografija", "dijagram", "mapa", "karta", "tabela", "grafikon"}

    if not aktivan_filter:
        return {"fotografija_profil", "dijagram", "mapa", "karta"}

    dozvoljeni = set()
    for kategorija, info in KATEGORIJA_MAPPING.items():
        if any(tv in aktivan_filter for tv in info["tip_values"]):
            dozvoljeni |= KAT_SLIKA.get(kategorija, set())
    return dozvoljeni

# ============================================================
# KEŠIRANJE SVIH ODLOMAKA IZ BAZE
# ============================================================
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

                    tip = r.payload.get("tip", "")

                    slika_url = (r.payload.get("Link") or r.payload.get("slika_url") or
                                 r.payload.get("image_url") or r.payload.get("slika") or
                                 r.payload.get("photo_url") or r.payload.get("url") or "")

                    if not slika_url and raw_txt:
                        img_match = RE_URL.search(raw_txt)
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
                            "slika_url": str(slika_url).strip(),
                            "tip": tip
                        })

            if next_offset is None or len(records) == 0:
                break

            offset = next_offset
    except Exception as e:
        st.warning(f"Upozorenje pri učitavanju baze: {e}")

    return sve_tacke

# ============================================================
# PRIKAZ SLIKA
# ============================================================
def filtriraj_slike_za_prikaz(top_k_stavke, upit, aktivan_filter=None, max_slika=6):
    """
    Bira koje slike/dijagrame da prikaže. Koristi KAT_SLIKA mapu kategorija
    → dozvoljeni tipovi slika da bi se izbacio "slučajni" vizuel iz drugog
    dokumenta u top-K. Korisnik može eksplicitno zatražiti sliku/dijagram
    rečima "prikaži", "pokaži" — tad se dozvoljavaju svi tipovi.
    """
    eksplicitno_vizuel = je_eksplicitno_vizuelni_upit(upit)
    dozvoljeni_tipovi = dozvoljeni_tipovi_za_filter(aktivan_filter, eksplicitno_vizuel)

    prikazi_slike = []
    vidjene = set()

    for item in top_k_stavke:
        tip = item.get("tip", "")
        if tip not in dozvoljeni_tipovi:
            continue

        url = item.get("slika_url", "").strip()
        if not url or not url.startswith("http") or url in vidjene:
            continue

        oznaka = "Fotografija" if tip == "fotografija_profil" else "Dijagram"
        prikazi_slike.append((url, f"{oznaka}. Izvor: {item['izvor']}"))
        vidjene.add(url)

        if len(prikazi_slike) >= max_slika:
            break

    return prikazi_slike

# ============================================================
# OPTIMIZOVANI HIBRIDNI PRETRAŽIVAČ (Stage 2)
# ------------------------------------------------------------
# Glavne novine u Stage 2:
#   1. E5 embedding (768 dim) sa "query: " prefiksom
#   2. Auto-detekcija kategorije iz upita → Qdrant filter na "tip"
#   3. Fallback bez filtera ako filtrirani rezultati daju premalo kandidata
#   4. Meta-podaci o aktivnom filteru prikazani korisniku
# ============================================================
def dobij_hibridni_kontekst(upit, top_k_rezultata=6, max_karaktera=6000, min_rezultata_sa_filterom=3):
    svi_odlomci = ucitaj_sve_tekstove()
    upit_ascii = ukloni_dijakritike(upit)
    norm_upit = sredi_tekst(upit)

    # --- NOVO: Auto-detekcija kategorije i pravljenje filtera ---
    tip_vrednosti = detektuj_kategoriju(upit)
    qdrant_filter = napravi_qdrant_filter(tip_vrednosti)
    aktivan_filter = tip_vrednosti  # Čuva se za prikaz u metapodacima

    candidates_map = {}
    koreni = izvuci_korene(upit_ascii)

    brojevi = re.findall(r'\b\d+\b', upit)
    je_clan_upit = any(w in upit_ascii for w in ["clan", "cl", "ugovor", "kolektivni", "ku"])
    clan_res = [re.compile(r'\b(?:clan|cl)[a-z]*\.?\s*' + re.escape(str(br)) + r'\b') for br in brojevi] if (brojevi and je_clan_upit) else []

    je_zamenik = "zamenik" in upit_ascii or "zamenici" in upit_ascii
    je_direktor = ("direktor" in upit_ascii or "rukovodilac" in upit_ascii) and not je_zamenik

    # --- LINEarni SKEN ---
    # Kad je filter aktivan, preskačemo odlomke čiji 'tip' NIJE u filter listi.
    # Zapisi BEZ 'tip' polja se NE preskaču (mogu biti relevantni), ali
    # dobijaju MANJI SCORE od zapisa sa matching tipom — tako da tip-kategorizovani
    # zapisi (npr. fotografije zaposlenih za kadrovski filter) uvek
    # rangiraju iznad nekategorizovanih (koji mogu biti nepovezani).
    for item in svi_odlomci:
        if aktivan_filter and item.get("tip") and item["tip"] not in aktivan_filter:
            continue
        # BEZ TIPA: propusti (možda je relevantno, nemamo metadata)

        txt_a = item["tekst_ascii"]
        izv_a = item["izvor_ascii"]
        key = item["tekst"]
        score = 0.0

        if clan_res:
            for cre in clan_res:
                if cre.search(txt_a):
                    score += 200000.0

        if je_direktor:
            if "zamenik" not in txt_a and ("direktor" in txt_a or "direktor" in izv_a):
                score += 150000.0
        elif je_zamenik:
            if any(w in txt_a for w in ["zamenik", "zamenici", "svetlana", "mihajlovic", "goran", "caldovic"]):
                score += 150000.0

        if koreni:
            tekst_pogodaka = sum(1 for k, skracen in koreni if koren_prisutan(k, skracen, txt_a))
            izvor_pogodaka = sum(1 for k, skracen in koreni if koren_prisutan(k, skracen, izv_a))
            ukupno = tekst_pogodaka * 5 + izvor_pogodaka
            if ukupno > 0:
                score += ukupno * 10000.0

        # JAK BOOST za matching tip kad je filter aktivan — ovo rešava haos
        # kod upita tipa "zaposleni" gde OSNOVA chunkovi (no tip, sa statistikama)
        # dobijaju isti score kao kadrovski zapisi. Sa boostom, pravi tip uvek
        # pobeduje.
        if aktivan_filter and item.get("tip") and item["tip"] in aktivan_filter:
            score += 500000.0

        if score > 0:
            candidates_map[key] = {"item": item, "score": score}

    # --- VEKTORSKA PRETRAGA (QDRANT) SA FILTEROM ---
    # Pokušavamo sa filterom prvo. Ako vrati premalo, ponavljamo bez filtera.
    vektor_pokusaji = [
        (qdrant_filter, "sa filterom"),
        (None, "bez filtera (fallback)"),
    ]
    vektor_rezultati = []
    koriscen_filter = qdrant_filter
    for filter_obj, _label in vektor_pokusaji:
        try:
            query_vector = embed_upit(norm_upit)
            points = qdrant.search(
                collection_name=COLLECTION_NAME,
                query_vector=query_vector,
                query_filter=filter_obj,
                limit=15
            )
            vektor_rezultati = points
            koriscen_filter = filter_obj
            # Ako smo već bez filtera ili imamo dovoljno, prekini
            if filter_obj is None or len(points) >= min_rezultata_sa_filterom:
                break
        except Exception:
            continue

    for rank, hit in enumerate(vektor_rezultati):
        if hit.payload:
            raw_txt = (hit.payload.get("tekst") or hit.payload.get("text") or hit.payload.get("content") or "")
            izvor = (hit.payload.get("naziv_dokumenta") or hit.payload.get("file_name") or hit.payload.get("izvor") or "")
            tip = hit.payload.get("tip", "")
            slika_url = (hit.payload.get("Link") or hit.payload.get("slika_url") or
                         hit.payload.get("image_url") or hit.payload.get("slika") or "")

            if raw_txt:
                norm_txt = sredi_tekst(raw_txt)
                vec_score = (15 - rank) * 100.0

                if norm_txt in candidates_map:
                    candidates_map[norm_txt]["score"] += vec_score
                else:
                    # NOVO: Čak i za nove kandidate iz vektor pretrage,
                    # poštuj aktivni filter (osim ako je u fallback režimu)
                    if aktivan_filter and koriscen_filter is not None and tip and tip not in aktivan_filter:
                        continue
                    candidates_map[norm_txt] = {
                        "item": {
                            "tekst": norm_txt,
                            "tekst_ascii": ukloni_dijakritike(norm_txt),
                            "izvor": sredi_tekst(izvor),
                            "izvor_ascii": ukloni_dijakritike(izvor),
                            "slika_url": str(slika_url).strip(),
                            "tip": tip
                        },
                        "score": vec_score
                    }

    if not candidates_map and svi_odlomci:
        for item in svi_odlomci[:5]:
            candidates_map[item["tekst"]] = {"item": item, "score": 10.0}

    # RANGIRANJE
    rangirani = sorted(candidates_map.values(), key=lambda x: x["score"], reverse=True)
    top_k = [entry["item"] for entry in rangirani[:top_k_rezultata]]

    # SLIKE — poštuju aktivni filter i eksplicitnost namere korisnika
    slike_za_prikaz = filtriraj_slike_za_prikaz(top_k, upit, aktivan_filter=tip_vrednosti)

    MAX_PO_ODLOMKU = 900
    kontekst_delovi = []
    for item in top_k:
        cist_txt = RE_CLEAN_URL.sub('', item["tekst"]).strip()
        if len(cist_txt) > MAX_PO_ODLOMKU:
            cist_txt = cist_txt[:MAX_PO_ODLOMKU] + "...[odlomak skraćen]"
        kontekst_delovi.append(f"Odlomak iz dokumenta [{item['izvor']}]:\n{cist_txt}")

    spojeni_tekst = "\n\n---\n\n".join(kontekst_delovi)
    if len(spojeni_tekst) > max_karaktera:
        spojeni_tekst = spojeni_tekst[:max_karaktera] + "\n...[Skraćeno]"

    # Vraćamo i info o aktivnom filteru da prikažemo korisniku
    return spojeni_tekst, len(rangirani), len(svi_odlomci), slike_za_prikaz, aktivan_filter, koriscen_filter is not None

# ============================================================
# STRIMOVANJE GROQ ODGOVORA
# ============================================================
def strimuj_groq_odgovor(poruke):
    try:
        response_stream = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=poruke,
            temperature=0.1,
            max_tokens=500,
            stream=True
        )
        for chunk in response_stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
    except Exception as e:
        err_str = str(e).lower()
        if "429" in err_str or "rate_limit" in err_str:
            st.toast("⚠️ Kratkotrajni limit aktiviran. Pravim pauzu od 3 sekunde...", icon="⏳")
            time.sleep(3)
            try:
                response_stream = groq_client.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    messages=poruke,
                    temperature=0.1,
                    max_tokens=500,
                    stream=True
                )
                for chunk in response_stream:
                    if chunk.choices and chunk.choices[0].delta.content:
                        yield chunk.choices[0].delta.content
            except Exception as e2:
                yield "\n\n⚠️ **Server je trenutno opterećen zahtevima. Molimo sačekajte 10-ak sekundi pa ponovite pitanje.**"
        else:
            raise e

# ============================================================
# BOČNI MENI
# ============================================================
with st.sidebar:
    st.image("https://pub-49fb3cc788a74e0a9edbac7e11305b94.r2.dev/srbijasume_logo.jpg", use_container_width=True)
    st.markdown(
        "<h1 style='text-align: center; color: #1b4332; margin-top: 8px;'>🌲 Биро асистент</h1>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='text-align: center; color: #1b4332;'><strong>Дигитални асистент Бироа за планирање</strong></p>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='text-align: center; color: #52796f; font-style: italic;'>ПД „Србијашуме”</p>",
        unsafe_allow_html=True,
    )
    st.divider()

    st.markdown("### 🛠️ Статус система")
    st.caption(f"🟢 **Векторска база:** Qdrant Cloud")
    st.caption(f"🟢 **Колекција:** `{COLLECTION_NAME}`")
    st.caption(f"🟢 **Embedding:** {EMBEDDING_MODEL.split('/')[-1]} ({EMBEDDING_DIM} dim)")
    st.caption(f"🟢 **Језички модел:** Groq Llama (са резервним)")
    st.caption(f"🟢 **Филтер по типу:** активан")

    st.divider()

    if st.button("🔄 Освежи кеш базе", use_container_width=True):
        st.cache_data.clear()
        st.success("Кеш је освежен!")

    if st.button("🧹 Обриши разговор", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ============================================================
# GLAVNO ZAGLAVLJE
# ============================================================
st.markdown("""
<div class="main-header">
    <img src="https://pub-49fb3cc788a74e0a9edbac7e11305b94.r2.dev/srbijasume_logo.jpg" style="height:90px;margin-bottom:12px;">
    <h1>🌲 Биро за планирање</h1>
    <p>ПД „Србијашуме” — Дигитални асистент</p>
</div>
""", unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.expander("💡 Брза предложена питања (кликните да поставите)", expanded=(len(st.session_state.messages) == 0)):
    col1, col2, col3, col4 = st.columns(4)
    clicked_prompt = None
    if col1.button("👔 Ко је директор?", use_container_width=True):
        clicked_prompt = "Ко је директор Бироа и покажи његову слику?"
    if col2.button("👥 Ко су заменици?", use_container_width=True):
        clicked_prompt = "Ко су заменици директора у Бироу и прикажи њихове слике?"
    if col3.button("🖨️ Шта имамо од опреме?", use_container_width=True):
        clicked_prompt = "Које штампаче и тонере користимо у Бироу?"
    if col4.button("📜 Чланови 14 и 18?", use_container_width=True):
        clicked_prompt = "Наведи члан 14 и члан 18 Колективног уговора."

    if clicked_prompt:
        st.session_state.prompt_input = clicked_prompt

# ============================================================
# PRIKAZ ISTORIJE PORUKA
# ============================================================
for msg in st.session_state.messages:
    avatar = "👤" if msg["role"] == "user" else "🌲"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])

        if "image_data" in msg and msg["image_data"]:
            for url, cap in msg["image_data"]:
                st.image(url, width=300, caption=cap)

# ============================================================
# OBRADA UNOSA KORISNIKA
# ============================================================
prompt = st.chat_input("Поставите питање...")

if "prompt_input" in st.session_state and st.session_state.prompt_input:
    prompt = st.session_state.prompt_input
    del st.session_state.prompt_input

if prompt:
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar="🌲"):
        with st.spinner("Претражујем базу и генеришем одговор..."):
            try:
                kontekst, br_kandidata, ukupno_keširano, slike_podaci, aktivan_filter, filter_primenjen = dobij_hibridni_kontekst(prompt)

                system_instruction = (
                    "Ti si stručni digitalni asistent Biroa za planiranje (PD Srbijašume).\n"
                    "Odgovaraj ISKLJUČIVO na osnovu dostavljenog KONTEKSTA.\n\n"
                    "STROGA PRAVILA ZA ODGOVARANJE:\n"
                    "1. FOKUSIRAJ SE NA SPECIFIČAN POJAM IZ PITANJA. Ako korisnik pita 'Koji toneri se koriste za štampače?', fokusiraj se SAMO na tonere — ne ponavljaj listu štampača. Za pitanja o široj kategoriji ('Koji su štampači?') daj opšte informacije. Za pitanja o USKOM pojmu unutar kategorije, daj SAMO informacije o tom pojmu.\n"
                    "2. KORISNIK MOŽE TRAŽITI SLIKU — TI SE U ODGOVORU UOPŠTE NE BAVI PRIKAZOM SLIKA (aplikacija to sama radi). NIKAD ne pominji, ne komentariši i ne izvinjavaj se za (ne)mogućnost prikazivanja slika kao digitalni asistent — jednostavno opiši sadržaj kao da je slika već prikazana pored tvog odgovora.\n"
                    "3. Ako se tražena osoba ili podatak NE NALAZI u dostavljenom kontekstu, kratko kaži da podatak nije pronađen.\n"
                    "4. ZABRANJENO JE nuditi druge osobe iz konteksta kao zamenu.\n"
                    "5. STROGO JE ZABRANJENO ispisivanje URL linkova ili slika u formatu Markdown.\n"
                    "6. SVAKO NOVO PITANJE dobija NOVI, SVEŽI KONTEKST iz baze (nalazi se uz 'Trenutno korisničko pitanje' ispod). UVEK odgovaraj na osnovu TOG novog konteksta — nikad ne prenosi informacije iz prethodnih odgovora u ovom razgovoru na novo, drugačije pitanje. Prethodne poruke koristi SAMO za razumevanje kratkih potpitanja tipa 'daj više detalja' ili 'a šta piše o tome' — u svim ostalim slučajevima ignorišti prethodnu temu.\n"
                    "7. TI NEMAŠ UVID U SAMU SLIKU/FOTOGRAFIJU, samo u tekstualni opis iz baze. NIKAD ne izmišljaj i ne pretpostavljaj kako slika izgleda (boje, izraz lica, odeća, kompozicija, 'verovatno prikazuje...') — prenesi SAMO činjenice koje stvarno piše u tekstu konteksta (ime, funkcija, naslov dokumenta), ništa vizuelno mimo toga.\n"
                    "8. AKO KORISNIK TRAŽI DIJAGRAM/SLIKU/ŠEMU, a u kontekstu NE POSTOJI vizuelni zapis sa odgovarajućim 'tip' poljem, eksplicitno reci da traženi vizuel NIJE pronađen. NIKAD nemoj pominjati druge dijagrame/slike/fotografije iz konteksta kao zamenu (čak i ako postoje). Aplikacija sama upravlja prikazom slika — tvoj posao je SAMO da preneseš šta STVARNO piše u kontekstu za dati upit.\n"
                    "9. KAD KORISNIK PITA O ZAPOSLENIMA U BIROU: Tvoj odgovor MORA biti lista KONKRETNIH OSOBA sa IMENIMA i FUNKCIJAMA. Ti podaci MORAJU biti izričito prisutni u kontekstu kao 'zaposleni u Birou'. Ako kontekst ne sadrži eksplicitnu listu zaposlenih, reci: 'Nemam eksplicitnu listu zaposlenih u Birou, ali imam <N> fotografija osoba iz baze.' NIKAD nemoj pominjati autore priručnika, naučne radove, autore knjiga, ili osobe koje se pominju u kontekstu ali NISU eksplicitno zaposlene u Birou. Ako kontekst sadrži statističke podatke o zaposlenosti (po opštinama, sektorima, godinama) — IGNORIŠI ih potpuno, to NIJE odgovor na pitanje o ljudima u Birou.\n"
                    "Odgovaraj isključivo na srpskom jeziku."
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

                with st.expander("🔍 Преглед метаподатака претраге"):
                    st.caption(f"Укупно одломака у кешу: **{ukupno_keširano}**")
                    st.caption(f"Рангираних кандидата: **{br_kandidata}**")
                    if slike_podaci:
                        st.caption(f"Приказана визуелна референца: {len(slike_podaci)}")
                    # NOVO: Prikaz aktivnog filtera
                    if aktivan_filter:
                        if filter_primenjen:
                            st.caption(f"🎯 **Активан филтер:** `{aktivan_filter}`")
                        else:
                            st.caption(f"⚠️ **Детектована категорија** (nema dovoljno rezultata sa filterom): `{aktivan_filter}` — fallback na celu bazu")
                    st.text_area("Прочишћен текстуални контекст послат моделу:", value=kontekst, height=200)

                st.session_state.messages.append({"role": "user", "content": prompt})
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": odgovor,
                    "image_data": slike_podaci
                })

            except Exception as e:
                st.error(f"Дошло је до грешке у комуникацији: {e}")
