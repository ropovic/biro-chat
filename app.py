"""
app.py — Biro za planiranje RAG asistent (Streamlit)
======================================================

Verzija 3.0 — fuzija svega:
- Embedding: paraphrase-multilingual-mpnet-base-v2 (768 dim)
- Qdrant filter po `tip` polju sa 6 kategorija
- Auto-filter: vizuel samo sa "prikaži/pokaži" ključnim rečima
- Linear scan + Qdrant hybrid retrieval
- Tip boost: +500K za matching tip
- Reranker (jina-reranker-v2-multilingual) — default ON lokalno, OFF na Streamlit Cloud
- 6 brzih pitanja u jednom redu
- "Lista zaposlenih" specijalni handler — vraća SVIH 25 sa slikama
- Bolji dijagram filter (ruža vetrova po ključnoj reči)
"""

import os
import re
import time
import uuid
import hashlib
import streamlit as st
from qdrant_client import QdrantClient, models
from groq import Groq
from fastembed import TextEmbedding

# ============================================================
# KONFIGURACIJA
# ============================================================
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
EMBEDDING_DIM = 768
COLLECTION_NAME = os.environ.get("COLLECTION_NAME", "baza_cloud_v2_e5")
R2_PUBLIC_URL = "https://pub-49fb3cc788a74e0a9edbac7e11305b94.r2.dev"
LOGO_URL = f"{R2_PUBLIC_URL}/srbijasume_logo.jpg"
# Reranker: isključen po defaultu (Streamlit Cloud 1 GB limit).
# LOKALNO: $env:USE_RERANKER="true" pre pokretanja.
USE_RERANKER = os.environ.get("USE_RERANKER", "false").lower() == "true"

SYSTEM_PROMPT = (
    "Ti si stručni digitalni asistent Biroa za planiranje (PD Srbijašume).\n"
    "Odgovaraj ISKLJUČIVO na osnovu dostavljenog KONTEKSTA.\n\n"
    "STROGA PRAVILA ZA ODGOVARANJE:\n"
    "1. FOKUSIRAJ SE NA SPECIFIČAN POJAM IZ PITANJA. Ako korisnik pita 'Koji toneri se koriste za štampače?', fokusiraj se SAMO na tonere — ne ponavljaj listu štampača.\n"
    "2. KORISNIK MOŽE TRAŽITI SLIKU — aplikacija to sama radi.\n"
    "3. Ako se tražena osoba ili podatak NE NALAZI u kontekstu, kratko kaži da podatak nije pronađen.\n"
    "4. ZABRANJENO JE nuditi druge osobe iz konteksta kao zamenu.\n"
    "5. STROGO JE ZABRANJENO ispisivanje URL linkova ili slika u Markdown formatu u tvom tekstualnom odgovoru — slike prikazuje aplikacija automatski.\n"
    "6. SVAKO NOVO PITANJE dobija NOVI KONTEKST — ne prenosi info iz prethodnih odgovora osim ako korisnik eksplicitno traži 'daj detalje o tome'.\n"
    "7. TI NEMAŠ UVID U SLIKU — samo u tekstualni opis. ZABRANJENO je opisivati boje ili detalje sa slike.\n"
    "8. AKO KORISNIK TRAŽI DIJAGRAM, a nema ga u kontekstu, eksplicitno reci da nije pronađen.\n"
    "9. KAD KORISNIK PITA O ZAPOSLENIMA: Sve osobe čija se IMENA pojavljuju u kontekstu smatraju se zaposlenima. NAVEDI SVA IMENA. Funkcije navedi SAMO ako su eksplicitno navedene u kontekstu. IGNORIŠI statističke podatke (procenat, broj zaposlenih po kategorijama, plate).\n"
    "10. AKO KONTEKST SADRŽI OCR-ED TEKST: INTERPRETIRAJ ga sažeto, daj suštinu — ne kopiraj sirovi OCR.\n"
    "Odgovaraj isključivo na srpskom jeziku."
)

# ============================================================
# KLIJENTI
# ============================================================
@st.cache_resource
def get_clients():
    qdrant = QdrantClient(
        url=os.environ["QDRANT_URL"],
        api_key=os.environ["QDRANT_API_KEY"],
        check_compatibility=False,
    )
    groq = Groq(api_key=os.environ["GROQ_API_KEY"])
    embed_model = TextEmbedding(model_name=EMBEDDING_MODEL)
    return qdrant, groq, embed_model


qdrant, groq_client, embed_model = get_clients()

# ============================================================
# NORMALIZACIJA + STEMOVANJE
# ============================================================
def sredi_tekst(tekst):
    if not tekst:
        return ""
    tekst = str(tekst)
    tekst = tekst.replace('Љ', 'Lj').replace('љ', 'lj').replace('Њ', 'Nj').replace('њ', 'nj').replace('Џ', 'Dž').replace('џ', 'dž')
    zamene = {
        'а': 'a', 'б': 'b', 'в': 'v', 'г': 'g', 'д': 'd', 'е': 'e',
        'ж': 'ž', 'з': 'z', 'и': 'i', 'ј': 'j', 'к': 'k', 'л': 'l',
        'м': 'm', 'н': 'n', 'о': 'o', 'п': 'p', 'р': 'r', 'с': 's', 'т': 't',
        'у': 'u', 'ф': 'f', 'х': 'h', 'ц': 'c', 'ч': 'č', 'ш': 'š',
    }
    return "".join([zamene.get(ch, ch) for ch in tekst])


def ukloni_dijakritike(tekst):
    if not tekst:
        return ""
    zamene = {'č': 'c', 'ć': 'c', 'š': 's', 'ž': 'z', 'đ': 'd'}
    return "".join([zamene.get(ch, ch) for ch in sredi_tekst(tekst).lower()])


def _stemuj_rec(w):
    if len(w) >= 7:
        return w[:-2]
    elif len(w) >= 5:
        return w[:-1]
    return w


STOP_RECI = {"ko", "je", "su", "sta", "pise", "bazi", "postoji", "navedi", "prikazi", "pokazi",
             "slika", "slike", "sliku", "foto", "fotografij", "u", "i", "na", "sa", "za", "o",
             "da", "li", "ima", "njegovu", "njihove", "njena", "mesto", "radno", "biro", "biroa",
             "planiranje", "projektovanje", "pd", "srbijasume", "sumarstvu", "detalje", "detaljnije",
             "koji", "koja", "koje", "kog", "kojoj", "kojim", "svi", "sve", "svih", "kao", "ali",
             "ili", "gde", "kada", "kako", "ovaj", "ova", "ovo", "taj", "ta", "to", "vec", "samo",
             "jos", "vrlo", "neki", "neka", "neko", "nesto", "moze", "molim", "mi", "vas", "ovo"}
STOP_KORENI = {_stemuj_rec(w) for w in STOP_RECI}


def izvuci_korene(upit_ascii):
    reci = [w for w in re.findall(r'\b\w+\b', upit_ascii) if len(w) > 2]
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


def embed_upit(tekst):
    if "e5" in EMBEDDING_MODEL.lower():
        tekst = f"query: {tekst}"
    return list(embed_model.embed([tekst]))[0].tolist()

# ============================================================
# KATEGORIJE I FILTERI
# ============================================================
KATEGORIJA_MAPPING = {
    "oprema": {
        "keywords": ["stampač", "stampac", "stampaci", "printer", "pisač", "pisac",
                     "toner", "toneri", "kertridž", "kertridz", "kartuša", "cartridge",
                     "skener", "monitor", "računar", "racunar", "kompjuter", "laptop",
                     "miš", "mis", "tastatura", "oprema", "inventar", "mreža", "mreza",
                     "štampa", "stampa", "server", "hardver", "uredjaj", "uređaj",
                     "kertridza", "tonera", "laserski", "laser", "inkjet"],
        "tip_values": ["oprema", "kancelarijska_oprema", "inventar", "toner", "stampac"],
    },
    "kadrovski": {
        "keywords": ["direktor", "rukovodilac", "zaposleni", "radnik", "šef", "sef",
                     "osoblje", "lice", "lica", "kadrovi", "imenik", "biografija",
                     "fotografija", "profil", "kadrovsk", "kadrovska", "kadrove",
                     "struktura", "lista", "spisak", "imen", "ljudi", "tim"],
        "tip_values": ["kadrovski", "zaposleni", "osoblje", "fotografija_profil", "biografija",
                       "kadrovska_struktura", "kadrovski_podaci"],
    },
    "pravni": {
        "keywords": ["član", "clan", "ugovor", "kolektivni", "pravilnik", "zakon",
                     "propis", "odluka", "rešenje", "resenje", "statut", "članovi"],
        "tip_values": ["pravni_akt", "ugovor", "kolektivni_ugovor", "pravilnik", "zakon", "odluka"],
    },
    "projektna": {
        "keywords": ["gazdinska", "g.j.", "gj ", "osnova", "projekat", "šuma", "sume",
                     "šumski", "sumski", "drvna", "drvno", "seča", "seca", "gazdinstv",
                     "gospodarska", "jedinica", "smola", "prirast", "panj", "sortiment",
                     "etat", "drvna masa", "krupno drvo", "sitno drvo", "celuloza",
                     "projektantsk", "revizij", "karta", "mapa", "oblast"],
        "tip_values": ["projektna_dokumentacija", "sumskoprivredna_osnova",
                       "gospodarska_jedinica", "mapa", "karta", "dijagram"],
    },
    "vizuel": {
        "keywords": ["dijagram", "grafikon", "šema", "shema", "mapa", "karta",
                     "ruža vetrova", "ruza vetrova", "vetrova", "ruža",
                     "crtež", "ilustracija", "skica", "prikaz", "tabela", "shema"],
        "tip_values": ["dijagram", "mapa", "karta", "tabela", "grafikon", "vizuel"],
    },
    "osoba": {
        "keywords": ["fotografija", "profil", "lica", "lice",
                     "izgled", "portret", "slika zaposlenog", "slika osobe"],
        "tip_values": ["fotografija_profil", "biografija"],
    },
}

KAT_SLIKA = {
    "vizuel":     {"dijagram", "mapa", "karta", "grafikon", "tabela"},
    "osoba":      {"fotografija_profil", "biografija"},
    "kadrovski":  {"fotografija_profil", "biografija"},
    "projektna":  {"mapa", "karta", "dijagram", "tabela"},
    "oprema":     set(),
    "pravni":     set(),
}

# Specifični filteri za dijagrame po ključnoj reči
DIJAGRAM_KLJUČNE_REČI = {
    "ruža vetrova": ["ruza_vetrova", "windrose", "vetrova", "ruža", "wind_rose"],
    "panj": ["panj", "panjeva", "panjevi", "stump"],
    "sortiment": ["sortiment", "sortimanata", "drvn", "drveta"],
    "prirast": ["prirast", "prirastaj"],
    "gazdinska": ["gazdinska", "g.j.", "gj ", "gazdinstv"],
    "karta": ["karta", "mapa"],
}


def je_eksplicitno_vizuelni_upit(upit):
    upit_lower = (upit or "").lower()
    return any(r in upit_lower for r in [
        "prikaži", "prikazi", "pokaži", "pokazi",
        "daj mi sliku", "daj sliku", "prikaži sliku",
        "prikaži fotografiju", "prikaži dijagram",
        "prikaži šemu", "prikaži mapu", "prikaži crtež",
        "prikazati", "pokazati",
    ])


def je_pitanje_o_zaposlenima(upit):
    """Detektuje da li korisnik traži listu zaposlenih."""
    upit_lower = (upit or "").lower()
    paterni = [
        r"\bko su (svi |svi\s+)?zaposleni",
        r"\blista zaposleni",
        r"\bspisak zaposleni",
        r"\bnavedi (sve )?zaposlene",
        r"\bsvi (u |iz )?biro",
        r"\bko (sve )?radi",
        r"\bimenik\b",
        r"\bko su ljudi",
        r"\bkadrovska struktura",
        r"\bstruktura zaposleni",
    ]
    return any(re.search(p, upit_lower) for p in paterni)


def detektuj_kategoriju(upit):
    upit_lower = upit.lower()
    eksplicitno_vizuel = je_eksplicitno_vizuelni_upit(upit)
    matches = set()
    for kategorija, info in KATEGORIJA_MAPPING.items():
        for kw in info["keywords"]:
            if kw in upit_lower:
                if kategorija == "vizuel" and not eksplicitno_vizuel:
                    break
                matches.update(info["tip_values"])
                break
    return list(matches) if matches else None


def napravi_qdrant_filter(tip_vrednosti):
    if not tip_vrednosti:
        return None
    return models.Filter(
        should=[models.FieldCondition(key="tip", match=models.MatchAny(any=tip_vrednosti))]
    )


def dozvoljeni_tipovi_za_filter(aktivan_filter, eksplicitno_vizuel=False):
    if eksplicitno_vizuel:
        return {"fotografija_profil", "biografija", "dijagram", "mapa", "karta", "tabela", "grafikon"}
    if not aktivan_filter:
        return {"fotografija_profil", "dijagram", "mapa", "karta"}
    dozvoljeni = set()
    for kategorija, info in KATEGORIJA_MAPPING.items():
        if any(tv in aktivan_filter for tv in info["tip_values"]):
            dozvoljeni |= KAT_SLIKA.get(kategorija, set())
    return dozvoljeni


def specificni_dijagram_tip(upit):
    """Detektuje specifičan tip dijagrama po ključnoj reči u upitu."""
    upit_lower = upit.lower()
    for kljucna_rec, tipovi in DIJAGRAM_KLJUČNE_REČI.items():
        if kljucna_rec in upit_lower:
            # Vrati listu mogućih tipova
            return tipovi
    return None


# ============================================================
# KADROVSKA LISTA — SVIH 25 ZAPOSLENIH
# ============================================================
def izvuci_imena_iz_teksta(tekst):
    """Izvlači moguća imena iz teksta."""
    if not tekst:
        return []
    # Normalizuj prvo
    norm = tekst
    # Srpska slova: tražimo uzorak "VelikoSlovo+ maloSlova"
    # Dozvoli č, ć, š, ž, đ
    pattern = r'\b([A-ZČĆŠĐŽ][a-zčćšđž]{2,}(?:\s+[A-ZČĆŠĐŽ][a-zčćšđž]{2,}){1,2})\b'
    matches = re.findall(pattern, norm)

    # Filtriranje — izbaci lažne pogotke
    blacklist = {
        "BiZa Planiranje", "Biro Za", "Srbija Šume", "Srbijasume", "Sumarstvo Srbije",
        "Šumarski Fakultet", "Beograd Sumarstvo", "Univerzitet U", "Ministarstvo Poljoprivrede",
        "Republika Srbija", "Grad Beograd", "Opstina Beograd", "Uprava Za", "Direkcija Za",
        "Kolektivni Ugovor", "Kadrovski Pravilnik", "Pravilnik O", "Statut Preduzeca",
        "Osnivacki Akt", "Sistematizacija Radnih", "Mesto Rada", "Radno Mesto",
        "Biro Za Planiranje", "Preduzece Za", "Sume Srbije", "Javno Preduzece",
        "Sumsko Privredna", "Osnova Gazdovanja", "Gazdinska Jedinica", "Gospodarska Jedinica",
        "Etat Sume", "Drvna Masa", "Krupno Drvo", "Sitno Drvo", "Celuloza I",
    }
    titles = {"dr", "mr", "prof", "doc", "ing", "inž", "dipl"}

    rezultat = []
    for m in matches:
        if m in blacklist:
            continue
        if any(b in m for b in ["Sumarstvo", "Beograd", "Srbija", "Sume", "Biro", "Fakultet",
                                 "Univerzitet", "Ministarstvo", "Pravilnik", "Ugovor", "Uredba",
                                 "Zakon", "Praviln"]):
            continue
        # Očisti od titula
        reci = m.split()
        ciste = [r for r in reci if r.lower().rstrip(".") not in titles]
        if ciste and len(ciste) >= 2:
            rezultat.append(" ".join(ciste))
    return rezultat


def get_svi_zaposleni():
    """Vraća listu svih zaposlenih sa fotografijama gde postoje."""
    points, _ = qdrant.scroll(
        collection_name=COLLECTION_NAME,
        scroll_filter=models.Filter(should=[
            models.FieldCondition(key="tip", match=models.MatchAny(any=[
                "kadrovski", "zaposleni", "osoblje", "kadrovska_struktura",
                "kadrovski_podaci", "fotografija_profil", "biografija",
            ]))
        ]),
        with_payload=True,
        with_vectors=False,
        limit=1000,
    )

    # Mapa: ključ (prezime) -> {name, photo, sources}
    imenik = {}

    for p in points:
        payload = p.payload or {}
        tip = payload.get("tip", "")
        text = payload.get("tekst", "") or payload.get("text", "")
        izvor = (payload.get("naziv_dokumenta", "") or payload.get("file_name", "") or
                 payload.get("izvor", "") or payload.get("dokument", "") or "")
        url = (payload.get("Link", "") or payload.get("slika_url", "") or
               payload.get("image_url", "") or payload.get("slika", ""))

        if "fotografija_profil" in tip or url:
            # Foto zapis — koristi tekst ili izvor za ime
            ime = None
            if text:
                imena = izvuci_imena_iz_teksta(text)
                if imena:
                    ime = imena[0]
            if not ime and izvor:
                # Probaj iz naziva fajla
                ime_iz_fajla = re.sub(r'\.(jpg|jpeg|png|webp)$', '', izvor, flags=re.IGNORECASE)
                ime_iz_fajla = ime_iz_fajla.replace("_", " ").strip()
                if ime_iz_fajla:
                    ime = ime_iz_fajla.title()
            if ime:
                kljuc = ime.lower().split()[-1]  # prezime
                if kljuc not in imenik:
                    imenik[kljuc] = {"ime": ime, "foto": url, "izvori": []}
                else:
                    imenik[kljuc]["foto"] = url or imenik[kljuc]["foto"]
                    if ime != imenik[kljuc]["ime"]:
                        # Ažuriraj ime ako je novije kompletnije
                        if len(ime) > len(imenik[kljuc]["ime"]):
                            imenik[kljuc]["ime"] = ime
        else:
            # Tekstualni zapis — traži imena
            if text:
                imena = izvuci_imena_iz_teksta(text)
                for ime in imena:
                    kljuc = ime.lower().split()[-1]
                    if kljuc not in imenik:
                        imenik[kljuc] = {"ime": ime, "foto": "", "izvori": []}
                    if izvor and izvor not in imenik[kljuc]["izvori"]:
                        imenik[kljuc]["izvori"].append(izvor)

    # Konverzija u listu, sortirano po prezimenu
    zaposleni = sorted(imenik.values(), key=lambda x: x["ime"].lower().split()[-1])
    return zaposleni


# ============================================================
# UČITAVANJE TEKSTOVA I RETRIEVAL
# ============================================================
def ucitaj_sve_tekstove():
    sve_tacke = []
    offset = None
    while True:
        records, next_offset = qdrant.scroll(
            collection_name=COLLECTION_NAME,
            limit=250,
            offset=offset,
            with_payload=True,
            with_vectors=False,
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
                if raw_txt:
                    sve_tacke.append({
                        "tekst": sredi_tekst(raw_txt),
                        "tekst_ascii": ukloni_dijakritike(sredi_tekst(raw_txt)),
                        "izvor": sredi_tekst(izvor),
                        "izvor_ascii": ukloni_dijakritike(sredi_tekst(izvor)),
                        "slika_url": str(slika_url).strip(),
                        "tip": tip,
                    })
        if next_offset is None or len(records) == 0:
            break
        offset = next_offset
    return sve_tacke


@st.cache_data(ttl=1800)
def get_tekstovi():
    return ucitaj_sve_tekstove()


def filtriraj_slike_za_prikaz(top_k_stavke, upit, aktivan_filter=None, max_slika=10):
    eksplicitno_vizuel = je_eksplicitno_vizuelni_upit(upit)
    dozvoljeni_tipovi = dozvoljeni_tipovi_za_filter(aktivan_filter, eksplicitno_vizuel)

    # Ako je specifičan dijagram (npr. "ruža vetrova"), filtriraj po izvoru
    specificni_tip = specificni_dijagram_tip(upit)
    if specificni_tip:
        dozvoljeni_tipovi = dozvoljeni_tipovi & {"dijagram", "mapa", "karta", "grafikon", "tabela"}

    prikazi_slike = []
    vidjene = set()
    for item in top_k_stavke:
        tip = item.get("tip", "")
        if tip not in dozvoljeni_tipovi:
            continue
        url = item.get("slika_url", "").strip()
        if not url or not url.startswith("http") or url in vidjene:
            continue

        # Specifični filter za dijagram
        if specificni_tip:
            izvor_a = item.get("izvor_ascii", "").lower()
            if not any(kw in izvor_a for kw in specificni_tip):
                continue

        oznaka = "Fotografija" if tip == "fotografija_profil" else "Dijagram/mapa"
        prikazi_slike.append((url, f"{oznaka}. Izvor: {item['izvor']}"))
        vidjene.add(url)
        if len(prikazi_slike) >= max_slika:
            break
    return prikazi_slike


# ============================================================
# RE-RANKER (Stage 2)
# ============================================================
_reranker = None


@st.cache_resource
def get_reranker():
    global _reranker
    if not USE_RERANKER:
        return None
    try:
        from fastembed.rerank.cross_encoder import TextCrossEncoder
        # Na Streamlit Cloud (1 GB) ovo može OOM-ovati
        # Pokreni lokalno sa $env:USE_RERANKER="true"
        _reranker = TextCrossEncoder(model_name="jinaai/jina-reranker-v2-base-multilingual")
        return _reranker
    except Exception as e:
        st.warning(f"Reranker load failed: {e}. Koristim linearni score.")
        return None


def rerank_candidates(query, candidates, top_n):
    if not candidates or len(candidates) <= top_n:
        return candidates[:top_n] if candidates else []
    reranker = get_reranker()
    if reranker is None:
        return candidates[:top_n]
    try:
        texts = [c.get("tekst", "") for c in candidates]
        pairs = [(query, t) for t in texts]
        scores = list(reranker.rerank(pairs))
        scored = list(zip(candidates, scores))

        def get_score(item):
            s = item[1]
            if hasattr(s, "score"):
                return float(s.score)
            if isinstance(s, (tuple, list)) and len(s) >= 2:
                return float(s[1])
            return 0.0

        scored.sort(key=get_score, reverse=True)
        return [c for c, _ in scored[:top_n]]
    except Exception:
        return candidates[:top_n]


# ============================================================
# GLAVNI RETRIEVAL
# ============================================================
def dobij_kontekst_i_slike(upit, top_k_rezultata=10, max_karaktera=8000):
    svi_odlomci = get_tekstovi()
    upit_ascii = ukloni_dijakritike(upit)
    norm_upit = sredi_tekst(upit)

    tip_vrednosti = detektuj_kategoriju(upit)
    qdrant_filter = napravi_qdrant_filter(tip_vrednosti)
    aktivan_filter = tip_vrednosti

    candidates_map = {}
    koreni = izvuci_korene(upit_ascii)

    # Linearni sken
    for item in svi_odlomci:
        if aktivan_filter and item.get("tip") and item["tip"] not in aktivan_filter:
            continue
        txt_a = item["tekst_ascii"]
        izv_a = item["izvor_ascii"]
        key = item["tekst"]
        score = 0.0
        if koreni:
            tekst_pogodaka = sum(1 for k, skracen in koreni if koren_prisutan(k, skracen, txt_a))
            izvor_pogodaka = sum(1 for k, skracen in koreni if koren_prisutan(k, skracen, izv_a))
            ukupno = tekst_pogodaka * 5 + izvor_pogodaka
            if ukupno > 0:
                score += ukupno * 10000.0
        if aktivan_filter and item.get("tip") and item["tip"] in aktivan_filter:
            score += 500000.0
        if score > 0:
            candidates_map[key] = {"item": item, "score": score}

    # Qdrant pretraga (sa filterom, pa bez ako nema dovoljno)
    vektor_rezultati = []
    koriscen_filter = qdrant_filter
    for filter_obj, _ in [(qdrant_filter, "sa filterom"), (None, "bez filtera")]:
        try:
            query_vector = embed_upit(norm_upit)
            points = qdrant.search(
                collection_name=COLLECTION_NAME,
                query_vector=query_vector,
                query_filter=filter_obj,
                limit=15,
            )
            vektor_rezultati = points
            koriscen_filter = filter_obj
            if filter_obj is None or len(points) >= 3:
                break
        except Exception:
            continue

    for rank, hit in enumerate(vektor_rezultati):
        if hit.payload:
            raw_txt = (hit.payload.get("tekst") or hit.payload.get("text") or
                       hit.payload.get("content") or "")
            izvor = (hit.payload.get("naziv_dokumenta") or hit.payload.get("file_name") or
                     hit.payload.get("izvor") or "")
            tip = hit.payload.get("tip", "")
            slika_url = (hit.payload.get("Link") or hit.payload.get("slika_url") or
                         hit.payload.get("image_url") or hit.payload.get("slika") or "")
            if raw_txt:
                norm_txt = sredi_tekst(raw_txt)
                vec_score = (15 - rank) * 100.0
                if norm_txt in candidates_map:
                    candidates_map[norm_txt]["score"] += vec_score
                else:
                    if aktivan_filter and koriscen_filter is not None and tip and tip not in aktivan_filter:
                        continue
                    candidates_map[norm_txt] = {
                        "item": {
                            "tekst": norm_txt,
                            "tekst_ascii": ukloni_dijakritike(norm_txt),
                            "izvor": sredi_tekst(izvor),
                            "izvor_ascii": ukloni_dijakritike(sredi_tekst(izvor)),
                            "slika_url": str(slika_url).strip(),
                            "tip": tip,
                        },
                        "score": vec_score,
                    }

    if not candidates_map and svi_odlomci:
        for item in svi_odlomci[:5]:
            candidates_map[item["tekst"]] = {"item": item, "score": 10.0}

    rangirani = sorted(candidates_map.values(), key=lambda x: x["score"], reverse=True)

    # Re-ranker
    if USE_RERANKER and len(rangirani) > top_k_rezultata:
        rerank_input = [{
            "tekst": e["item"].get("tekst", ""),
            "item": e["item"],
            "score": e["score"],
        } for e in rangirani[:30]]
        reranked = rerank_candidates(norm_upit, rerank_input, top_k_rezultata)
        top_k = [r["item"] for r in reranked]
    else:
        top_k = [e["item"] for e in rangirani[:top_k_rezultata]]

    # Kontekst za LLM
    MAX_PO_ODLOMKU = 900
    kontekst_delovi = []
    for item in top_k:
        cist_txt = re.sub(r'http[s]?://\S+', '', item["tekst"]).strip()
        if len(cist_txt) > MAX_PO_ODLOMKU:
            cist_txt = cist_txt[:MAX_PO_ODLOMKU] + "...[odlomak skraćen]"
        kontekst_delovi.append(f"Odlomak iz dokumenta [{item['izvor']}]:\n{cist_txt}")
    spojeni = "\n\n---\n\n".join(kontekst_delovi)
    if len(spojeni) > max_karaktera:
        spojeni = spojeni[:max_karaktera] + "\n...[Skraćeno]"

    slike = filtriraj_slike_za_prikaz(top_k, upit, aktivan_filter=aktivan_filter)
    return spojeni, len(rangirani), len(svi_odlomci), slike, aktivan_filter, koriscen_filter is not None


# ============================================================
# LLM POZIV
# ============================================================
def pitaj_llm(poruke):
    try:
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=poruke,
            temperature=0.1,
            max_tokens=600,
            stream=False,
        )
        return response.choices[0].message.content
    except Exception as e:
        err_str = str(e).lower()
        if "429" in err_str or "rate_limit" in err_str:
            try:
                response = groq_client.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    messages=poruke,
                    temperature=0.1,
                    max_tokens=600,
                )
                return response.choices[0].message.content
            except Exception:
                return "⚠️ Server je trenutno opterećen. Pokušajte ponovo za 10 sekundi."
        return f"⚠️ Greška: {e}"


# ============================================================
# STREAMLIT UI
# ============================================================
st.set_page_config(
    page_title="Биро асистент",
    page_icon="🌲",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Header
col_l, col_c, col_r = st.columns([1, 3, 1])
with col_c:
    st.image(LOGO_URL, width=110)
    st.markdown(
        "<h1 style='text-align: center; color: #1b4332; margin-top: 0;'>🌲 Биро асистент</h1>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='text-align: center; color: #52796f; margin-top: -10px;'>"
        "ПД „Србијашуме” • Биро за планирање</p>",
        unsafe_allow_html=True,
    )

st.markdown("---")

# 6 dugmadi u jednom redu
st.markdown("##### 💡 Брза питања:")
quick_cols = st.columns(6)
QUICK_PROMPTS = [
    "👤 Директор Бироа",
    "👥 Сви запослени",
    "🖨️ Штампачи у Бироу",
    "🎨 Тонери",
    "🌀 Ружа ветрова",
    "📜 Члан 14",
]
for i, label in enumerate(QUICK_PROMPTS):
    with quick_cols[i]:
        if st.button(label, use_container_width=True, key=f"quick_{i}"):
            st.session_state.pending_question = label

st.markdown("---")

# Status
with st.sidebar:
    st.image(LOGO_URL, width=90)
    st.markdown("### 🌲 Биро асистент")
    st.caption(f"Kolekcija: `{COLLECTION_NAME}`")
    st.caption(f"Model: {EMBEDDING_MODEL.split('/')[-1]}")
    st.caption(f"Reranker: {'✅ ON' if USE_RERANKER else '❌ OFF (1 GB limit)'}")
    st.caption(f"Baza: {get_tekstovi().__len__() if get_tekstovi() else 0} zapisa")
    st.markdown("---")
    st.markdown("**Legenda:**")
    st.markdown("👤 = osoba  \n🖨️ = oprema  \n📜 = pravni  \n🌲 = projektna  \n🌀 = dijagram")

# Istorija chata
if "messages" not in st.session_state:
    st.session_state.messages = []

# Prikaz prethodnih poruka
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("images"):
            cols = st.columns(min(4, len(msg["images"])))
            for idx, (url, cap) in enumerate(msg["images"]):
                with cols[idx % 4]:
                    st.image(url, caption=cap, use_container_width=True)

# Input
user_input = st.chat_input("Поставите питање...")
if "pending_question" in st.session_state:
    user_input = st.session_state.pending_question
    del st.session_state.pending_question

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Тражим у бази..."):
            try:
                # Specijalni handler za listu zaposlenih
                if je_pitanje_o_zaposlenima(user_input):
                    zaposleni = get_svi_zaposleni()
                    if zaposleni:
                        imena_lista = [z["ime"] for z in zaposleni]
                        # Prikaz imena
                        st.markdown(f"### 👥 Запослени у Бироу ({len(zaposleni)})")
                        st.markdown("\n".join([f"- **{ime}**" for ime in imena_lista]))

                        # Galerija slika
                        sa_sl = [z for z in zaposleni if z.get("foto")]
                        if sa_sl:
                            st.markdown("---")
                            st.markdown(f"#### 📸 Фотографије ({len(sa_sl)}/{len(zaposleni)})")
                            cols = st.columns(min(4, len(sa_sl)))
                            for idx, z in enumerate(sa_sl):
                                with cols[idx % 4]:
                                    st.image(z["foto"], caption=z["ime"], use_container_width=True)
                    else:
                        st.markdown("⚠️ Nisu pronađeni zaposleni u bazi.")

                    final_response = f"Pronađeno **{len(zaposleni)}** zaposlenih."
                    st.session_state.messages.append({
                        "role": "assistant", "content": final_response, "images": []
                    })
                else:
                    # Standardni RAG
                    kontekst, br_kandidata, ukupno, slike, aktivan_filter, filter_primenjen = \
                        dobij_kontekst_i_slike(user_input)

                    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
                    for h in st.session_state.messages[-4:-1]:
                        if h["role"] != "system":
                            messages.append({"role": h["role"], "content": h["content"]})

                    upit = f"KONTEKST IZ BAZE:\n{kontekst}\n\nTrenutno korisničko pitanje: {user_input}"
                    messages.append({"role": "user", "content": upit})

                    odgovor = pitaj_llm(messages)

                    if slike:
                        st.markdown("### 📎 Vizuelne reference")
                        cols = st.columns(min(4, len(slike)))
                        for idx, (url, cap) in enumerate(slike):
                            with cols[idx % 4]:
                                st.image(url, caption=cap, use_container_width=True)
                        st.markdown("---")

                    st.markdown(odgovor)

                    meta = (
                        f"<sub>📊 Kandidati: {br_kandidata} | Baza: {ukupno} | "
                        f"Filter: {aktivan_filter or 'nema'} | Slike: {len(slike)}</sub>"
                    )
                    if aktivan_filter and not filter_primenjen:
                        meta += " <sub>⚠️ fallback</sub>"
                    st.markdown(meta, unsafe_allow_html=True)

                    st.session_state.messages.append({
                        "role": "assistant", "content": odgovor, "images": slike
                    })
            except Exception as e:
                st.error(f"⚠️ Greška: {e}")
                st.session_state.messages.append({
                    "role": "assistant", "content": f"⚠️ Greška: {e}", "images": []
                })
