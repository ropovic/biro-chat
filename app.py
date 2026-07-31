"""
app.py — Biro za planiranje RAG asistent (Streamlit)
======================================================

Verzija 4.0 — Pametni ruteri (NE SVE ide kroz LLM):
- "Ko je direktor?" → direktno iz baze, BEZ LLM halucinacije
- "Svi zaposleni" → listanje + slike, BEZ LLM
- "Ima li Bojane?" → pretraga po imenu, BEZ LLM
- "Štampači/toneri" → direktan prikaz, BEZ LLM
- Ostalo → LLM sa strogim promptom

Performance:
- Bez rerankera (1 GB limit)
- TTL keša 1200 (20 min)
- max_karaktera 5000 (umesto 8000) — brži LLM
- max_tokens 400 (umesto 600) — brži LLM
"""

import os
import re
import time
import streamlit as st
from qdrant_client import QdrantClient, models
from groq import Groq
from fastembed import TextEmbedding

# ============================================================
# KONFIG
# ============================================================
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
COLLECTION_NAME = os.environ.get("COLLECTION_NAME", "baza_cloud_v2_e5")
R2_PUBLIC_URL = "https://pub-49fb3cc788a74e0a9edbac7e11305b94.r2.dev"
LOGO_URL = f"{R2_PUBLIC_URL}/srbijasume_logo.jpg"

SYSTEM_PROMPT = (
    "Ti si stručni digitalni asistent Biroa za planiranje (PD Srbijašume).\n"
    "Odgovaraj ISKLJUČIVO na osnovu dostavljenog KONTEKSTA.\n\n"
    "KRITIČNA PRAVILA:\n"
    "1. NEMOJ IZVOĐITI ZAKLJUČKE. Ako kontekst ne pominje TAČNO traženi podatak, "
    "odgovori: 'Podatak nije pronađen u bazi.' NE izmišljaj, NE pretpostavljaj.\n"
    "2. KAD KORISNIK PITA 'Ko je direktor?' — navedi SAMO ako kontekst EKSPLICITNO kaže "
    "'direktor je Ime Prezime'. Ako se ime pojavljuje BEZ eksplicitne funkcije, "
    "reci 'nije eksplicitno navedeno ko je direktor'.\n"
    "3. KAD KORISNIK PITA O ZAPOSLENIMA — navedi SAMO imena koja se pojavljuju kao "
    "LIČNA IMENA. IGNORIŠI opise organizacione strukture (odseci, službe, pozicije).\n"
    "4. NEMOJ generisati URL-ove ni Markdown slike — aplikacija to radi automatski.\n"
    "5. Odgovaraj isključivo na srpskom jeziku. Kratko i jasno.\n"
    "6. Ako kontekst sadrži OCR — interpretiraj sažeto, NE kopiraj sirovi tekst.\n"
    "7. Ako kontekst ima 'Ime Prezime' (placeholder) — IGNORIŠI taj zapis.\n"
    "8. Ako je pitanje nejasno, traži pojašnjenje umesto da nagađaš."
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
# NORMALIZACIJA
# ============================================================
def sredi_tekst(t):
    if not t:
        return ""
    t = str(t).replace('Љ', 'Lj').replace('љ', 'lj').replace('Њ', 'Nj').replace('њ', 'nj')
    zamene = {'а':'a','б':'b','в':'v','г':'g','д':'d','е':'e','ж':'ž','з':'z','и':'i',
              'ј':'j','к':'k','л':'l','м':'m','н':'n','о':'o','п':'p','р':'r','с':'s',
              'т':'t','у':'u','ф':'f','х':'h','ц':'c','ч':'č','ш':'š'}
    return "".join([zamene.get(c, c) for c in t])


def ukloni_dijakritike(t):
    if not t:
        return ""
    zamene = {'č':'c','ć':'c','š':'s','ž':'z','đ':'d'}
    return "".join([zamene.get(c, c) for c in sredi_tekst(t).lower()])


def embed_upit(tekst):
    if "e5" in EMBEDDING_MODEL.lower():
        tekst = f"query: {tekst}"
    return list(embed_model.embed([tekst]))[0].tolist()


# ============================================================
# IMENA — Izdvajanje i blacklist
# ============================================================
# Organizacije, institucije, pozicije — NE SMEJU da prođu kao imena
IMENA_BLACKLIST = {
    # Organizacije
    "BiZa Planiranje", "Biro Za", "Biro Za Planiranje", "Srbija Šume", "Srbijasume",
    "Sumarstvo Srbije", "Sume Srbije", "Javno Preduzece", "Preduzece Za",
    "Šumarski Fakultet", "Beograd Sumarstvo", "Univerzitet U", "Ministarstvo Poljoprivrede",
    "Republika Srbija", "Grad Beograd", "Opstina Beograd", "Uprava Za", "Direkcija Za",
    "Kolektivni Ugovor", "Kadrovski Pravilnik", "Pravilnik O", "Statut Preduzeca",
    "Osnivacki Akt", "Sistematizacija Radnih", "Mesto Rada", "Radno Mesto",
    "Sumsko Privredna", "Osnova Gazdovanja", "Gazdinska Jedinica", "Gospodarska Jedinica",
    "Etat Sume", "Drvna Masa", "Krupno Drvo", "Sitno Drvo", "Celuloza I",
    "Šumsko Gazdinsko", "Šumskog Gazdinskog", "Upravno Odjeljenje", "Upravnog Odjeljenja",
    "Šef Šumskog", "Šef Upravnog", "Šumari Inženjeri", "Inženjeri Šumarstva",
    "Tehničari Šumarstva", "Tehnicari Šumarstva", "Šumski Radnici", "Šumskih Radnika",
    "Šumski Tehničar", "Šumski Inženjer", "Inženjer Šumarstva", "Tehničar Šumarstva",
    "Pomoćnik Direktora", "Pomoćnik Rukovodioca", "Rukovodilac Biroa",
    "Sekretar Biroa", "Administrator Biroa", "Voditelj Poslova",
    "Odjeljenje Za", "Sluzba Za", "Odsjek Za", "Sektor Za", "Referent Za",
    "Šef Službe", "Šef Odsjeka", "Šef Sektora", "Rukovodilac Sektora",
    "Šumsko Privredno", "Šumsko Gospodarsko", "Privredno Društvo", "Drustvo Za",
    "Ljudski Resursi", "Kadrovska Služba", "Pravna Služba", "Finansijska Služba",
    "PD Srbijašume", "PD Srbijasume", "JP Srbijasume",
}

# Reči koje NISU imena (pozicije, opisi)
NIJE_IME_KLJUCNE = [
    "Sumarstvo", "Beograd", "Srbija", "Sume", "Biro", "Fakultet", "Univerzitet",
    "Ministarstvo", "Pravilnik", "Ugovor", "Uredba", "Zakon", "Praviln",
    "Odjeljenje", "Odjeljenja", "Služba", "Odsjek", "Sektor", "Društvo", "Preduzece",
    "Resursi", "Uprava", "Upravno", "Gazdinsk", "Gazdinska", "Privred", "Radnik",
    "Radnika", "Tehničar", "Tehnicar", "Inženjer", "Inzenjer", "Šumari", "Sumari",
    "Šumsko", "Sumsko", "Rukovodilac", "Direktor", "Pomoćnik", "Pomoćnik",
    "Šef", "Sekretar", "Administrator", "Voditelj", "Referent", "Službenik",
    "Odsek", "Odseka", "Centar", "Centru", "Centra",
]

TITULE = {"dr", "mr", "prof", "doc", "ing", "inž", "dipl"}


def izvuci_imena_iz_teksta(tekst):
    """Izvlači SAMO validna lična imena iz teksta."""
    if not tekst:
        return []
    norm = tekst
    pattern = r'\b([A-ZČĆŠĐŽ][a-zčćšđž]{2,}(?:\s+[A-ZČĆŠĐŽ][a-zčćšđž]{2,}){1,2})\b'
    matches = re.findall(pattern, norm)

    rezultat = []
    for m in matches:
        # Tačno poklapanje sa blacklistom
        if m in IMENA_BLACKLIST:
            continue
        # Sadrži ključne reči (pozicije, organizacije)
        if any(b in m for b in NIJE_IME_KLJUCNE):
            continue
        # Očisti od titula
        reci = [r for r in m.split() if r.lower().rstrip(".") not in TITULE]
        if not reci or len(reci) < 2:
            continue
        # SVE reči moraju početi velikim slovom
        if not all(r[0].isupper() for r in reci if r[0].isalpha()):
            continue
        # Nema placeholder "Ime Prezime"
        if "Ime" in reci and "Prezime" in reci:
            continue
        # Minimalna dužina svake reči
        if any(len(r) < 3 for r in reci):
            continue
        # Filtriraj očigledno engleske/strane reči
        if any(r.lower() in {"the", "and", "for", "with", "from", "this", "that"} for r in reci):
            continue
        rezultat.append(" ".join(reci))
    return rezultat


def pronadji_osobu_po_imenu(ime_upita):
    """Traži specifičnu osobu po imenu. Vraća listu pogodaka sa slikama."""
    ime_lower = ime_upita.lower().strip()
    # Skroluj sve kadrovske/foto/biografija zapise
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
    pogodci = []
    for p in points:
        text = (p.payload.get("tekst", "") or "").lower()
        izvor = p.payload.get("naziv_dokumenta", "") or p.payload.get("file_name", "")
        url = (p.payload.get("Link", "") or p.payload.get("slika_url", "") or
               p.payload.get("image_url", "") or p.payload.get("slika", ""))
        # Da li je ime u tekstu?
        if ime_lower in text:
            # Izvuci originalno ime (sa velikim slovima)
            imena = izvuci_imena_iz_teksta(p.payload.get("tekst", ""))
            for ime in imena:
                if ime.lower() == ime_lower or ime.lower().split()[-1] == ime_lower.split()[-1]:
                    pogodci.append({
                        "ime": ime,
                        "izvor": izvor,
                        "url": url,
                        "kontekst": (p.payload.get("tekst", "") or "")[:300],
                    })
                    break
    # Dedupe po imenu
    seen = set()
    uniq = []
    for p in pogodci:
        if p["ime"].lower() not in seen:
            seen.add(p["ime"].lower())
            uniq.append(p)
    return uniq


def get_svi_zaposleni_sa_slikama():
    """Direktno iz baze: svi zaposleni + slike gde postoje."""
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
    imenik = {}
    for p in points:
        payload = p.payload or {}
        text = payload.get("tekst", "") or ""
        izvor = (payload.get("naziv_dokumenta", "") or payload.get("file_name", "") or "")
        url = (payload.get("Link", "") or payload.get("slika_url", "") or
               payload.get("image_url", "") or payload.get("slika", ""))
        # Izvuci imena iz teksta
        imena = izvuci_imena_iz_teksta(text)
        for ime in imena:
            kljuc = ime.lower()
            if kljuc not in imenik:
                imenik[kljuc] = {"ime": ime, "foto": "", "izvori": []}
            if url and not imenik[kljuc]["foto"]:
                imenik[kljuc]["foto"] = url
            if izvor and izvor not in imenik[kljuc]["izvori"]:
                imenik[kljuc]["izvori"].append(izvor)
    return sorted(imenik.values(), key=lambda x: x["ime"].split()[-1])


# ============================================================
# RUTIRANJE
# ============================================================
def detektuj_tip_upita(upit):
    """Vraća tip upita: 'direktor', 'osoba_ime', 'lista_zaposlenih', 'oprema', 'standard'."""
    u = upit.lower()
    # Direktor
    if re.search(r'\b(ko je|ko je novi|tko je)\b.*\bdirektor', u) or \
       re.search(r'\bdirektor\s+(biro| biroa| biroa za)', u) or \
       re.search(r'\bkako se zove direktor', u):
        return "direktor"
    # Lista zaposlenih
    if re.search(r'\b(svi |lista |spisak |navedi )?(zaposleni|zaposlene|svi u biro|svi iz biro|ko radi|kadrovska strukt|ljudi)\b', u) or \
       re.search(r'\bimenik\b', u):
        return "lista_zaposlenih"
    # Konkretna osoba (Ime Prezime ili samo ime)
    # Tražimo veliko slovo + još jedno veliko slovo u upitu
    imena_u_upitu = re.findall(r'\b[A-ZČĆŠĐŽ][a-zčćšđž]{2,}\s+[A-ZČĆŠĐŽ][a-zčćšđž]{2,}\b', upit)
    if imena_u_upitu:
        return "osoba_ime"
    # Oprema
    if re.search(r'\b(stampač|stampac|printer|toner|kertridž|oprema|štampač)\b', u):
        return "oprema"
    return "standard"


# ============================================================
# HANDLERI
# ============================================================
def handle_direktor():
    """Direktno traži direktora u bazi, BEZ LLM-a."""
    points, _ = qdrant.scroll(
        collection_name=COLLECTION_NAME,
        scroll_filter=models.Filter(should=[
            models.FieldCondition(key="tip", match=models.MatchAny(any=[
                "kadrovski", "zaposleni", "osoblje", "biografija",
            ]))
        ]),
        with_payload=True,
        with_vectors=False,
        limit=1000,
    )
    direktori = []
    for p in points:
        text = p.payload.get("tekst", "") or ""
        izvor = p.payload.get("naziv_dokumenta", "") or ""
        text_lower = text.lower()
        # Ako tekst eksplicitno kaže "direktor [Ime Prezime]"
        if re.search(r'\bdirektor[,\s]+[A-ZČĆŠĐŽ]', text):
            imena = izvuci_imena_iz_teksta(text)
            for ime in imena:
                if ime not in [d["ime"] for d in direktori]:
                    direktori.append({"ime": ime, "izvor": izvor})
    if not direktori:
        return "⚠️ **Nije eksplicitno pronađeno** ko je direktor Biroa u bazi. " \
               "Moguće je da ta informacija nedostaje u indeksiranim dokumentima.", []
    if len(direktori) == 1:
        d = direktori[0]
        # Pokušaj naći sliku direktora
        slike = []
        # Probaj pronaći fotografiju za to ime
        foto_pogodci = pronadji_osobu_po_imenu(d["ime"])
        for fp in foto_pogodci[:2]:
            if fp["url"]:
                slike.append((fp["url"], f"Direktor: {fp['ime']}"))
        return f"Direktor Biroa za planiranje je: **{d['ime']}**", slike
    imena = ", ".join([d["ime"] for d in direktori])
    return f"Prema bazi, direktori su: {imena}", []


def handle_lista_zaposlenih():
    """Vraća listu svih zaposlenih sa slikama, BEZ LLM-a."""
    zaposleni = get_svi_zaposleni_sa_slikama()
    if not zaposleni:
        return "⚠️ Nisu pronađeni zaposleni u bazi.", []
    imena_lista = "\n".join([f"- **{z['ime']}**" for z in zaposleni])
    sa_sl = [z for z in zaposleni if z.get("foto")]
    return (f"### 👥 Запослени у Бироу ({len(zaposleni)})\n\n{imena_lista}\n\n"
            f"📸 Фотографије: {len(sa_sl)}/{len(zaposleni)}"), [(z["foto"], z["ime"]) for z in sa_sl]


def handle_osoba_po_imenu(upit):
    """Traži osobu po imenu iz upita, BEZ LLM-a."""
    imena = re.findall(r'\b[A-ZČĆŠĐŽ][a-zčćšđž]{2,}\s+[A-ZČĆŠĐŽ][a-zčćšđž]{2,}\b', upit)
    if not imena:
        return "⚠️ Nisam pronašao ime u pitanju.", []
    svi_pogoci = []
    for ime in imena:
        pogoci = pronadji_osobu_po_imenu(ime)
        svi_pogoci.extend(pogoci)
    if not svi_pogoci:
        return f"⚠️ **{', '.join(imena)}** — nije pronađena u bazi.", []
    # Formatiraj
    delovi = []
    slike = []
    seen = set()
    for p in svi_pogoci:
        if p["ime"].lower() in seen:
            continue
        seen.add(p["ime"].lower())
        delovi.append(f"**{p['ime']}** — {p['izvor']}")
        if p["url"]:
            slike.append((p["url"], p["ime"]))
    return "### 👤 Pronađeno:\n\n" + "\n\n".join(delovi), slike[:6]


def handle_oprema(upit):
    """Direktno prikaži opremu, BEZ LLM-a za listanje."""
    u = upit.lower()
    if "toner" in u or "kertridž" in u:
        tip_filter = ["toner", "oprema"]
    else:
        tip_filter = ["oprema", "kancelarijska_oprema", "inventar", "stampac"]
    points = qdrant.search(
        collection_name=COLLECTION_NAME,
        query_vector=embed_upit(sredi_tekst(upit)),
        query_filter=models.Filter(should=[
            models.FieldCondition(key="tip", match=models.MatchAny(any=tip_filter))
        ]),
        limit=10,
    )
    if not points:
        return "⚠️ Nema pronađene opreme po tom upitu.", []
    # Prikaz liste bez LLM-a
    delovi = []
    for hit in points:
        text = hit.payload.get("tekst", "") or ""
        izvor = hit.payload.get("naziv_dokumenta", "") or ""
        # Skrati
        if len(text) > 400:
            text = text[:400] + "..."
        delovi.append(f"**[{izvor}]**\n{text}")
    return "### 🖨️ Pronađena oprema:\n\n" + "\n\n---\n\n".join(delovi), []


def handle_standard(upit, istorija):
    """Standardni RAG sa LLM-om i strogim promptom."""
    kontekst, br_kandidata, ukupno, slike = standardni_rag(upit)
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    # Istorija (samo prethodna 2 razgovora)
    for h in istorija[-2:]:
        if h.get("role") in ("user", "assistant"):
            messages.append({"role": h["role"], "content": h["content"]})
    messages.append({
        "role": "user",
        "content": f"KONTEKST:\n{kontekst}\n\nPitanje: {upit}"
    })
    try:
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            temperature=0.1,
            max_tokens=400,
        )
        odgovor = response.choices[0].message.content
    except Exception:
        # Fallback na manji model
        try:
            response = groq_client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=messages,
                temperature=0.1,
                max_tokens=400,
            )
            odgovor = response.choices[0].message.content
        except Exception as e:
            odgovor = f"⚠️ Greška: {e}"
    return odgovor, slike, br_kandidata, ukupno


# ============================================================
# STANDARDNI RAG
# ============================================================
@st.cache_data(ttl=1200)
def get_tekstovi():
    """Skroluje sve tekstove iz baze."""
    sve_tacke = []
    offset = None
    while True:
        records, next_offset = qdrant.scroll(
            collection_name=COLLECTION_NAME,
            limit=500,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        for r in records:
            if r.payload:
                raw_txt = (r.payload.get("tekst") or r.payload.get("text") or
                           r.payload.get("content") or "")
                izvor = (r.payload.get("naziv_dokumenta") or r.payload.get("file_name") or
                         r.payload.get("izvor") or "")
                tip = r.payload.get("tip", "")
                slika_url = (r.payload.get("Link") or r.payload.get("slika_url") or
                             r.payload.get("image_url") or "")
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


def _stemuj(w):
    if len(w) >= 7: return w[:-2]
    if len(w) >= 5: return w[:-1]
    return w


STOP = {"ko","je","su","sta","pise","bazi","postoji","navedi","prikazi","pokazi",
        "slika","slike","sliku","foto","u","i","na","sa","za","o","da","li","ima",
        "mesto","radno","biro","biroa","planiranje","projektovanje","pd","srbijasume",
        "detalje","detaljnije","koji","koja","koje","svi","sve","svih","kao","ali",
        "ili","gde","kada","kako","ovaj","ova","ovo","taj","ta","to","samo","jos",
        "vrlo","neki","neka","neko","nesto","moze","molim","mi","vas","ovo","sta",
        "trazi","trazim","nalaze","nadjem","naci","reci","kaze","kazu","neka","vec"}


def izvuci_korene(upit_ascii):
    reci = [w for w in re.findall(r'\b\w+\b', upit_ascii) if len(w) > 2]
    out = []
    for w in reci:
        k = _stemuj(w)
        if k in {_stemuj(s) for s in STOP}:
            continue
        out.append((k, len(k) < len(w)))
    return out


def koren_prisutan(k, skracen, tekst):
    if skracen:
        return re.search(r'\b' + re.escape(k), tekst) is not None
    return re.search(r'\b' + re.escape(k) + r'\b', tekst) is not None


def standardni_rag(upit, top_k=8, max_karaktera=5000):
    """Standardni RAG za opšta pitanja."""
    svi_odlomci = get_tekstovi()
    upit_ascii = ukloni_dijakritike(upit)
    norm_upit = sredi_tekst(upit)

    koreni = izvuci_korene(upit_ascii)
    candidates_map = {}

    # Linearni sken sa korenima
    for item in svi_odlomci:
        txt_a = item["tekst_ascii"]
        izv_a = item["izvor_ascii"]
        key = item["tekst"]
        score = 0.0
        if koreni:
            tp = sum(1 for k, s in koreni if koren_prisutan(k, s, txt_a))
            ip = sum(1 for k, s in koreni if koren_prisutan(k, s, izv_a))
            score = (tp * 5 + ip) * 10000.0
        if score > 0:
            candidates_map[key] = {"item": item, "score": score}

    # Qdrant semantička pretraga
    try:
        query_vector = embed_upit(norm_upit)
        points = qdrant.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_vector,
            limit=20,
        )
        for rank, hit in enumerate(points):
            if hit.payload:
                raw_txt = hit.payload.get("tekst", "") or hit.payload.get("text", "")
                izvor = (hit.payload.get("naziv_dokumenta", "") or
                         hit.payload.get("file_name", "") or "")
                tip = hit.payload.get("tip", "")
                slika_url = (hit.payload.get("Link", "") or hit.payload.get("slika_url", "") or "")
                if raw_txt:
                    norm_txt = sredi_tekst(raw_txt)
                    vec_score = (20 - rank) * 50.0
                    if norm_txt in candidates_map:
                        candidates_map[norm_txt]["score"] += vec_score
                    else:
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
    except Exception:
        pass

    rangirani = sorted(candidates_map.values(), key=lambda x: x["score"], reverse=True)
    top_k_items = [e["item"] for e in rangirani[:top_k]]

    # Kontekst
    MAX_PO = 700
    delovi = []
    for item in top_k_items:
        cist = re.sub(r'http[s]?://\S+', '', item["tekst"]).strip()
        if "Ime Prezime" in cist:
            continue  # Ignoriši placeholder
        if len(cist) > MAX_PO:
            cist = cist[:MAX_PO] + "..."
        delovi.append(f"[{item['izvor']}]\n{cist}")
    kontekst = "\n\n---\n\n".join(delovi)
    if len(kontekst) > max_karaktera:
        kontekst = kontekst[:max_karaktera] + "\n[Skraćeno]"

    # Slike — uzmi do 4
    slike = []
    seen = set()
    for item in top_k_items:
        url = item.get("slika_url", "").strip()
        if not url or not url.startswith("http") or url in seen:
            continue
        slike.append((url, f"Izvor: {item['izvor']}"))
        seen.add(url)
        if len(slike) >= 4:
            break

    return kontekst, len(rangirani), len(svi_odlomci), slike


# ============================================================
# UI
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

# 6 dugmadi
st.markdown("##### 💡 Брза питања:")
quick_cols = st.columns(6)
QUICK = [
    "👤 Директор",
    "👥 Сви запослени",
    "🖨️ Штампачи",
    "🎨 Тонери",
    "🌀 Ружа ветрова",
    "📜 Члан 14",
]
for i, label in enumerate(QUICK):
    with quick_cols[i]:
        if st.button(label, use_container_width=True, key=f"q{i}"):
            st.session_state.pending = label

st.markdown("---")

# Sidebar
with st.sidebar:
    st.image(LOGO_URL, width=90)
    st.markdown("### 🌲 Биро асистент")
    st.caption(f"Model: {EMBEDDING_MODEL.split('/')[-1]}")
    st.markdown("---")
    if st.button("🧹 Obriši razgovor", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    if st.button("🔄 Osveži bazu", use_container_width=True):
        get_tekstovi.clear()
        st.cache_data.clear()
        st.success("✅ Keš obrisan.")
        st.rerun()

# Istorija
if "messages" not in st.session_state:
    st.session_state.messages = []

# Prikaz
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("images"):
            cols = st.columns(min(3, len(msg["images"])))
            for idx, (url, cap) in enumerate(msg["images"]):
                with cols[idx % 3]:
                    st.image(url, caption=cap, use_container_width=True)

# Input
user_input = st.chat_input("Поставите питање...")
if "pending" in st.session_state:
    user_input = st.session_state.pending
    del st.session_state.pending

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Тражим..."):
            try:
                tip_upita = detektuj_tip_upita(user_input)
                slike = []
                meta = ""

                if tip_upita == "direktor":
                    odgovor, slike = handle_direktor()
                elif tip_upita == "lista_zaposlenih":
                    odgovor, slike = handle_lista_zaposlenih()
                elif tip_upita == "osoba_ime":
                    odgovor, slike = handle_osoba_po_imenu(user_input)
                elif tip_upita == "oprema":
                    odgovor, slike = handle_oprema(user_input)
                else:
                    odgovor, slike, br_k, ukupno = handle_standard(
                        user_input, st.session_state.messages
                    )
                    meta = f"\n\n<sub>📊 Kandidati: {br_k} | Baza: {ukupno}</sub>"

                # Prikaz slika
                if slike:
                    st.markdown("---")
                    cols = st.columns(min(3, len(slike)))
                    for idx, (url, cap) in enumerate(slike):
                        with cols[idx % 3]:
                            st.image(url, caption=cap, use_container_width=True)
                    st.markdown("---")

                st.markdown(odgovor + meta)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": odgovor + meta,
                    "images": slike,
                })
            except Exception as e:
                st.error(f"⚠️ Greška: {e}")
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"⚠️ Greška: {e}",
                    "images": [],
                })
