"""
app.py v20 — Poboljšano izvlačenje imena, veći limit za član, debug
"""

import os
import re
import time
import uuid
import streamlit as st
from qdrant_client import QdrantClient, models
from groq import Groq
from fastembed import TextEmbedding
import requests
import base64
import io
from PIL import Image

# ============================================================
# CONFIG
# ============================================================
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
COLLECTION_NAME = os.environ.get("COLLECTION_NAME", "baza_cloud_v5")
R2_PUBLIC_URL = os.environ.get("R2_PUBLIC_URL", "https://pub-49fb3cc788a74e0a9edbac7e11305b94.r2.dev")
LOGO_URL = f"{R2_PUBLIC_URL}/srbijasume_logo.jpg"

SYSTEM_PROMPT = (
    "Ti si digitalni asistent Biroa za planiranje (PD Srbijašume). "
    "Odgovaraj ISKLJUČIVO na osnovu KONTEKSTA. "
    "Ako podatak nije u kontekstu, reci 'nije pronađeno'. "
    "Fokusiraj se na specifičan pojam iz pitanja. "
    "Odgovaraj na srpskom, kratko i jasno."
)

# ============================================================
# CLIENTS
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
# POMOĆNE FUNKCIJE
# ============================================================
def embed_query(text):
    if "e5" in EMBEDDING_MODEL.lower():
        text = f"query: {text}"
    return list(embed_model.embed([text]))[0].tolist()

def qdrant_search(collection_name, query_vector, limit=10, query_filter=None, with_payload=True):
    if hasattr(qdrant, 'search'):
        return qdrant.search(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=limit,
            query_filter=query_filter,
            with_payload=with_payload,
        )
    elif hasattr(qdrant, 'query_points'):
        response = qdrant.query_points(
            collection_name=collection_name,
            query=query_vector,
            limit=limit,
            query_filter=query_filter,
            with_payload=with_payload,
        )
        return response.points
    else:
        raise Exception("Qdrant client nema ni 'search' ni 'query_points' metodu.")

def scroll_tip(tip, limit=10000):
    svi = []
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
            if r.payload and r.payload.get("tip") == tip:
                svi.append(r)
        if next_offset is None or len(records) == 0:
            break
        offset = next_offset
        if len(svi) >= limit:
            break
    return svi

def scroll_all(limit=10000):
    svi = []
    offset = None
    while True:
        records, next_offset = qdrant.scroll(
            collection_name=COLLECTION_NAME,
            limit=500,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        svi.extend(records)
        if next_offset is None or len(records) == 0:
            break
        offset = next_offset
        if len(svi) >= limit:
            break
    return svi

def search_text(query, top_k=10):
    vec = embed_query(query)
    try:
        return qdrant_search(
            collection_name=COLLECTION_NAME,
            query_vector=vec,
            limit=top_k,
            query_filter=None,
            with_payload=True,
        )
    except Exception as e:
        st.error(f"Greška pri pretrazi teksta: {e}")
        return []

# ============================================================
# PROŠIRENO IZVLAČENJE IMENA IZ TEKSTA (v2)
# ============================================================
def extract_name_from_text(tekst):
    """
    Pokušava da izvuče ime i prezime iz teksta na više načina.
    Vraća None ako ne može da izvuče.
    """
    if not tekst:
        return None

    # 1. "Zaposleni u Birou ...: Bojana Jelić"
    m = re.search(r'Zaposleni\s+u\s+Birou[^:]*:\s*([A-ZČĆŠĐŽ][a-zčćšđž]+\s+[A-ZČĆŠĐŽ][a-zčćšđž]+)', tekst, re.IGNORECASE)
    if m:
        return m.group(1)

    # 2. "Fotografija Bojana Jelić"
    m = re.search(r'[Ff]otografij[ae]\s+([A-ZČĆŠĐŽ][a-zčćšđž]+\s+[A-ZČĆŠĐŽ][a-zčćšđž]+)', tekst)
    if m:
        return m.group(1)

    # 3. Dve kapitalizovane reči na kraju (sa ili bez prefiksa)
    m = re.search(r'([A-ZČĆŠĐŽ][a-zčćšđž]+\s+[A-ZČĆŠĐŽ][a-zčćšđž]+)\s*$', tekst)
    if m:
        return m.group(1)

    # 4. Bilo koje dve kapitalizovane reči bilo gde (ali ne ako su deo "PD Srbijašume" i sl.)
    m = re.search(r'\b([A-ZČĆŠĐŽ][a-zčćšđž]+\s+[A-ZČĆŠĐŽ][a-zčćšđž]+)\b', tekst)
    if m:
        # Proveri da nije "PD Srbijašume" ili slično
        if m.group(1) not in ["PD Srbijašume", "Srbijašume"]:
            return m.group(1)

    # 5. Ćirilica: "Запослени у Бироу ...: Бојана Јелић"
    m = re.search(r'[Зз]апослени\s+у\s+Бироу[^:]*:\s*([А-ЯЁ][а-яё]+\s+[А-ЯЁ][а-яё]+)', tekst)
    if m:
        return m.group(1)

    # 6. Ćirilica: "Фотографија Бојана Јелић"
    m = re.search(r'[Фф]отографиј[ае]\s+([А-ЯЁ][а-яё]+\s+[А-ЯЁ][а-яё]+)', tekst)
    if m:
        return m.group(1)

    # 7. Ćirilica: dve kapitalizovane reči bilo gde
    m = re.search(r'\b([А-ЯЁ][а-яё]+\s+[А-ЯЁ][а-яё]+)\b', tekst)
    if m:
        return m.group(1)

    # 8. FALLBACK: uzmi prvu kapitalizovanu reč (ako je duža od 2 slova)
    # Ovo je opasno ali korisno za slučajeve gde ime nije u očekivanom formatu
    m = re.search(r'\b([A-ZČĆŠĐŽА-ЯЁ][a-zčćšđžа-яё]{2,})\b', tekst)
    if m:
        first = m.group(1)
        rest = tekst[m.end():]
        m2 = re.search(r'\b([A-ZČĆŠĐŽА-ЯЁ][a-zčćšđžа-яё]{2,})\b', rest)
        if m2:
            return f"{first} {m2.group(1)}"
        # Ako ima samo jedno ime, vrati ga (možda je ime bez prezimena)
        return first

    return None

# ============================================================
# HANDLERI
# ============================================================

def handle_direktor():
    points = scroll_tip("fotografija_profil")
    direktori = []
    zamenici = []

    def is_direktor(text):
        if not text:
            return False
        text_lower = text.lower()
        return "direktor" in text_lower or "директор" in text_lower

    def is_zamenik(text):
        if not text:
            return False
        text_lower = text.lower()
        return "zamenik" in text_lower or "заменик" in text_lower

    for p in points:
        payload = p.payload or {}
        tekst = payload.get("tekst", "") or ""
        url = payload.get("slika_url", "") or payload.get("Link", "")
        ime = payload.get("employee_name", "")
        if not ime:
            ime = extract_name_from_text(tekst)
        if not ime:
            continue

        funkcija = payload.get("Funkcija", "")
        if not funkcija:
            m = re.search(r'(?:Funkcija|funkcija|Функција)\s*[:;]\s*([^,.\n]+)', tekst, re.IGNORECASE)
            if m:
                funkcija = m.group(1).strip()

        if is_direktor(funkcija) and not is_zamenik(funkcija):
            direktori.append((ime, url, funkcija))
        elif is_zamenik(funkcija):
            zamenici.append((ime, url, funkcija))
        elif is_direktor(tekst) and not is_zamenik(tekst):
            direktori.append((ime, url, funkcija))
        elif is_zamenik(tekst):
            zamenici.append((ime, url, funkcija))

    if not direktori and not zamenici:
        hits = search_text("direktor", top_k=10)
        for hit in hits:
            payload = hit.payload or {}
            tekst = payload.get("tekst", "")
            if is_direktor(tekst):
                m = re.search(r'direktor\s+([A-ZČĆŠĐŽ][a-zčćšđž]+\s+[A-ZČĆŠĐŽ][a-zčćšđž]+)', tekst, re.IGNORECASE)
                if not m:
                    m = re.search(r'директор\s+([А-ЯЁ][а-яё]+\s+[А-ЯЁ][а-яё]+)', tekst)
                if m:
                    ime = m.group(1)
                    return f"**Direktor Biroa:** {ime}", []

    if not direktori and not zamenici:
        return "⚠️ Nisu pronađeni direktor/zamenici u bazi.", []

    slike = []
    delovi = []
    if direktori:
        ime = direktori[0][0]
        url = direktori[0][1]
        delovi.append(f"**Direktor Biroa:** {ime}")
        if url:
            slike.append((url, f"Direktor: {ime}"))
    if zamenici:
        ime = zamenici[0][0]
        url = zamenici[0][1]
        delovi.append(f"**Zamenik direktora:** {ime}")
        if url:
            slike.append((url, f"Zamenik: {ime}"))
        if len(zamenici) > 1:
            ostali = ", ".join([z[0] for z in zamenici[1:]])
            delovi.append(f"Ostali zamenici: {ostali}")

    return "\n\n".join(delovi), slike


def handle_lista_zaposlenih():
    all_points = scroll_all()
    zaposleni = []
    debug_texts = []  # za prikaz u sidebar-u

    for p in all_points:
        payload = p.payload or {}
        if payload.get("tip") != "fotografija_profil":
            continue
        tekst = payload.get("tekst", "") or ""
        url = payload.get("slika_url", "") or payload.get("Link", "")

        # Prvo probaj employee_name
        ime = payload.get("employee_name", "")
        if not ime:
            ime = extract_name_from_text(tekst)
            if not ime:
                # Ako nije uspeo, sačuvaj tekst za debug
                debug_texts.append(tekst[:100])
        if ime:
            funkcija = payload.get("Funkcija", "")
            if not funkcija:
                m = re.search(r'(?:Funkcija|funkcija|Функција)\s*[:;]\s*([^,.\n]+)', tekst, re.IGNORECASE)
                if m:
                    funkcija = m.group(1).strip()
            if not any(z["ime"].lower() == ime.lower() for z in zaposleni):
                zaposleni.append({"ime": ime, "url": url, "funkcija": funkcija})

    # Prikaži debug info u sidebar-u
    st.sidebar.write(f"📸 Ukupno fotografija u bazi: {len([p for p in all_points if p.payload.get('tip') == 'fotografija_profil'])}")
    st.sidebar.write(f"✅ Uspešno izvučeno imena: {len(zaposleni)}")
    if debug_texts:
        st.sidebar.write("⚠️ Tekstovi iz kojih NIJE izvučeno ime (prvih 5):")
        for t in debug_texts[:5]:
            st.sidebar.write(f"- {t}...")

    if not zaposleni:
        return "⚠️ Nisu pronađeni zaposleni.", []

    zaposleni.sort(key=lambda x: x["ime"])
    imena_lista = "\n".join([f"• {z['ime']} — {z['funkcija']}" for z in zaposleni if z.get("funkcija")])
    if not imena_lista:
        imena_lista = "\n".join([f"• {z['ime']}" for z in zaposleni])

    slike = [(z["url"], z["ime"]) for z in zaposleni if z.get("url")]
    return f"**Zaposleni u Birou ({len(zaposleni)}):**\n\n{imena_lista}", slike


def handle_osoba_po_imenu(upit):
    if not upit:
        return "⚠️ Prazno pitanje.", []

    words = upit.strip().split()
    stop_words = {"biro", "biroa", "srbijasume", "srbija", "suma", "birou", "kolektivni", "ugovor",
                  "clan", "preduzece", "firma", "kompanija", "pd", "jp", "svi", "sve", "sva",
                  "kako", "sta", "koji", "koja", "koje", "gde", "kada", "imam", "imaju",
                  "postoji", "treba", "hocu", "ovaj", "taj", "ovo", "ta", "to", "neka",
                  "neko", "nesto", "moze", "molim", "zasto", "zbog", "prema", "preko",
                  "ima", "li", "dal", "je", "da"}
    clean_words = [w for w in words if w.lower() not in stop_words and len(w) > 2]
    if not clean_words:
        return "⚠️ Nisam pronašao ime u pitanju.", []

    search_term = " ".join(clean_words)
    search_term_lower = search_term.lower()

    all_points = scroll_all()
    pogodci = []
    for p in all_points:
        payload = p.payload or {}
        if payload.get("tip") != "fotografija_profil":
            continue
        tekst = payload.get("tekst", "") or ""
        employee_name = payload.get("employee_name", "") or ""
        url = payload.get("slika_url", "") or payload.get("Link", "")

        if employee_name and search_term_lower in employee_name.lower():
            ime = employee_name
            funkcija = payload.get("Funkcija", "")
            if not funkcija:
                m = re.search(r'(?:Funkcija|funkcija|Функција)\s*[:;]\s*([^,.\n]+)', tekst, re.IGNORECASE)
                if m:
                    funkcija = m.group(1).strip()
            if not any(p["ime"].lower() == ime.lower() for p in pogodci):
                pogodci.append({"ime": ime, "url": url, "funkcija": funkcija})
            continue

        if search_term_lower in tekst.lower():
            ime = extract_name_from_text(tekst)
            if ime:
                funkcija = payload.get("Funkcija", "")
                if not funkcija:
                    m = re.search(r'(?:Funkcija|funkcija|Функција)\s*[:;]\s*([^,.\n]+)', tekst, re.IGNORECASE)
                    if m:
                        funkcija = m.group(1).strip()
                if not any(p["ime"].lower() == ime.lower() for p in pogodci):
                    pogodci.append({"ime": ime, "url": url, "funkcija": funkcija})

    if not pogodci:
        return f"⚠️ **{search_term}** — nisam pronašao u bazi fotografija.", []

    pogodci.sort(key=lambda x: x["ime"])
    delovi_text = [f"**Pronađeno {len(pogodci)}:**", ""]
    for p in pogodci:
        if p.get("funkcija"):
            delovi_text.append(f"- **{p['ime']}** — {p['funkcija']}")
        else:
            delovi_text.append(f"- **{p['ime']}**")
    slike = [(p["url"], p["ime"]) for p in pogodci if p["url"]][:5]
    return "\n".join(delovi_text), slike


def handle_oprema_specificno(upit):
    points = scroll_tip("oprema")
    u = sredi_upit(upit)
    is_toner = "toner" in u or "kertridz" in u
    is_printer = any(kw in u for kw in ["stampac", "stampaci", "printer", "pisac"])

    printer_pat = re.compile(
        r'\b(?:'
        r'Kyocera\s+(?:FS-\d+[a-z]*|M\d+[a-z]*|P\d+[a-z]*|TASKalfa\w*|ECOSYS\w*)'
        r'|HP\s+(?:Designjet\w*|OfficeJet\w*|LaserJet\w*|PageWide\w*)'
        r'|Canon\s+(?:imageRUNNER\w*|TX-\d+|iR\d+\w*|PIXMA\w*|imagePRESS\w*)'
        r'|Brother\s+(?:MFC\w*|DCP\w*|HL\w*)'
        r'|Epson\s+(?:WorkForce\w*|EcoTank\w*|SureColor\w*)'
        r')\b',
        re.IGNORECASE
    )

    toner_pat = re.compile(
        r'\b(?:'
        r'TK-\d{2,5}[A-Z]?'
        r'|HP\s+[CP]\d{3,5}[A-Z]?'
        r'|HP\s+CE\d{2,4}[A-Z]?'
        r'|Canon\s+(?:PFI-\d+\w*|CL-\d+\w*|PGI-\d+\w*)'
        r')\b',
        re.IGNORECASE
    )

    stampaci_lista = set()
    toneri_lista = set()

    for p in points:
        payload = p.payload or {}
        tekst_orig = payload.get("tekst", "") or ""
        if is_printer:
            for m in printer_pat.findall(tekst_orig):
                stampaci_lista.add(m.strip())
        elif is_toner:
            for m in toner_pat.findall(tekst_orig):
                toneri_lista.add(m.strip())
        else:
            for m in printer_pat.findall(tekst_orig):
                stampaci_lista.add(m.strip())
            for m in toner_pat.findall(tekst_orig):
                toneri_lista.add(m.strip())

    def dodaj_proizvodjaca(kod):
        if kod.upper().startswith("TK-"):
            return f"Kyocera {kod}"
        if kod.upper().startswith("CE") or kod.upper().startswith("HP"):
            return f"HP {kod}"
        return kod

    if is_toner:
        if not toneri_lista:
            return "⚠️ Nisu pronađeni toneri po tom upitu.", []
        delovi = ["**Toneri u Birou:**", ""]
        for t in sorted(toneri_lista):
            delovi.append(f"- {dodaj_proizvodjaca(t)}")
        return "\n".join(delovi), []

    if is_printer:
        if not stampaci_lista:
            return "⚠️ Nisu pronađeni štampači po tom upitu.", []
        delovi = ["**Štampači u Birou:**", ""]
        for s in sorted(stampaci_lista):
            delovi.append(f"- {s}")
        return "\n".join(delovi), []

    delovi = ["**Oprema u Birou:**", ""]
    if stampaci_lista:
        delovi.append("**Štampači:**")
        for s in sorted(stampaci_lista):
            delovi.append(f"- {s}")
        delovi.append("")
    if toneri_lista:
        delovi.append("**Toneri:**")
        for t in sorted(toneri_lista):
            delovi.append(f"- {dodaj_proizvodjaca(t)}")
    if not stampaci_lista and not toneri_lista:
        return "⚠️ Nema pronađene opreme.", []
    return "\n".join(delovi), []


def handle_dijagram(upit):
    # Prvo probaj pretragu teksta
    hits = search_text(upit, top_k=20)
    diag_hits = [h for h in hits if h.payload and h.payload.get("tip") in ["dijagram", "image"]]
    if diag_hits:
        slike = []
        for hit in diag_hits[:6]:
            payload = hit.payload
            url = payload.get("slika_url", "") or payload.get("Link", "")
            if url and url.startswith("http"):
                naziv = payload.get("naziv_dokumenta", "") or payload.get("izvor", "") or "Dijagram"
                slike.append((url, naziv))
        if slike:
            return f"**Pronađeno {len(slike)} slika/dijagrama:**", slike[:6]

    # Ako nema, uzmi sve dijagrame i filtriraj
    points = scroll_tip("dijagram")
    if not points:
        points = scroll_tip("image")
    if not points:
        return "⚠️ Nema pronađenih dijagrama/slika za ovaj upit.", []

    upit_lower = upit.lower()
    relevant = []
    for p in points:
        tekst = p.payload.get("tekst", "").lower()
        if upit_lower in tekst:
            relevant.append(p)
    if not relevant:
        relevant = points[:6]

    slike = []
    for p in relevant[:6]:
        url = p.payload.get("slika_url", "") or p.payload.get("Link", "")
        if url and url.startswith("http"):
            naziv = p.payload.get("naziv_dokumenta", "") or "Dijagram"
            slike.append((url, naziv))
    if not slike:
        return "⚠️ Pronađene slike nemaju javni URL.", []
    return f"**Pronađeno {len(slike)} slika/dijagrama:**", slike[:6]


def handle_clan(broj):
    for term in [f"član {broj}", f"члан {broj}"]:
        hits = search_text(term, top_k=20)
        for hit in hits:
            payload = hit.payload or {}
            tekst = payload.get("tekst", "")
            if re.search(rf'(?:član|clan|члан)\s*{re.escape(broj)}[\s.,;:)]', tekst, re.IGNORECASE):
                clan_pat = re.compile(
                    rf'(?:član|clan|члан)\s*{re.escape(broj)}[\s.,;:)](.*?)'
                    rf'(?=(?:član|clan|члан)\s*\d+[\s.,;:)]|$)',
                    re.DOTALL | re.IGNORECASE
                )
                m = clan_pat.search(tekst)
                if m:
                    clan_tekst = m.group(0).strip()
                    if len(clan_tekst) > 12000:  # povećano na 12000
                        clan_tekst = clan_tekst[:12000] + "\n...[Skraćeno]"
                    izvor = payload.get("naziv_dokumenta", "") or payload.get("izvor", "")
                    return f"**Član {broj}** (izvor: {izvor}):\n\n{clan_tekst}", []

    all_points = scroll_all()
    pogodci = []
    clan_pat = re.compile(
        rf'(?:član|clan|члан)\s*{re.escape(broj)}[\s.,;:)](.*?)'
        rf'(?=(?:član|clan|члан)\s*\d+[\s.,;:)]|$)',
        re.DOTALL | re.IGNORECASE
    )
    clan_check = re.compile(
        rf'(?:član|clan|члан)\s*{re.escape(broj)}[\s.,;:)]',
        re.IGNORECASE
    )

    for p in all_points:
        payload = p.payload or {}
        tekst = payload.get("tekst", "") or ""
        if not clan_check.search(tekst):
            continue
        for m in clan_pat.finditer(tekst):
            clan_tekst = m.group(0).strip()
            if len(clan_tekst) < 30:
                continue
            if len(clan_tekst) > 12000:
                clan_tekst = clan_tekst[:12000] + "\n...[Skraćeno]"
            izvor = payload.get("naziv_dokumenta", "") or payload.get("izvor", "")
            pogodci.append({"tekst": clan_tekst, "izvor": izvor, "duzina": len(clan_tekst)})
        if not pogodci:
            for m in clan_check.finditer(tekst):
                start = max(0, m.start() - 30)
                end = min(len(tekst), m.end() + 3500)
                clan_tekst = tekst[start:end].strip()
                if len(clan_tekst) >= 30:
                    pogodci.append({
                        "tekst": clan_tekst,
                        "izvor": payload.get("naziv_dokumenta", "") or payload.get("izvor", ""),
                        "duzina": len(clan_tekst),
                    })
                    break

    if not pogodci:
        return f"⚠️ Član {broj} nije pronađen u bazi.", []

    seen = set()
    uniq = []
    for p in pogodci:
        kljuc = sredi_upit(p["tekst"][:200])
        if kljuc not in seen:
            seen.add(kljuc)
            uniq.append(p)
    uniq.sort(key=lambda x: -x["duzina"])
    p = uniq[0]
    return f"**Član {broj}** (izvor: {p['izvor']}):\n\n{p['tekst']}", []


# ============================================================
# RAG
# ============================================================
def do_rag(query, top_k=10):
    try:
        hits = search_text(query, top_k=top_k)
    except Exception as e:
        return "", 0, [], f"Greška: {e}"

    delovi = []
    slike = []
    seen = set()
    for hit in hits:
        if not hit.payload:
            continue
        text = hit.payload.get("tekst", "") or ""
        izvor = hit.payload.get("naziv_dokumenta", "") or hit.payload.get("izvor", "") or ""
        url = hit.payload.get("slika_url", "") or hit.payload.get("Link", "")
        cist = re.sub(r'http[s]?://\S+', '', text).strip()
        cist = re.sub(r'\n{3,}', '\n\n', cist)
        if len(cist) > 500:
            cist = cist[:500] + "..."
        if cist:
            delovi.append(f"[{izvor}]\n{cist}")
        if url and url.startswith("http") and url not in seen:
            slike.append((url, izvor or "Slika"))
            seen.add(url)

    kontekst = "\n\n---\n\n".join(delovi)
    if len(kontekst) > 6000:
        kontekst = kontekst[:6000] + "\n[Skraćeno]"
    return kontekst, len(hits), slike, ""


def ask_llm(messages):
    try:
        resp = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            temperature=0.1,
            max_tokens=500,
        )
        return resp.choices[0].message.content
    except Exception:
        try:
            resp = groq_client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=messages,
                temperature=0.1,
                max_tokens=500,
            )
            return resp.choices[0].message.content
        except Exception as e:
            return f"⚠️ Greška: {e}"


# ============================================================
# EKSTERNA PRETRAGA
# ============================================================
import requests as req

def external_search(query, max_results=3):
    api_key = os.environ.get("TAVILY_API_KEY", "")
    if not api_key:
        return None
    try:
        r = req.post(
            "https://api.tavily.com/search",
            json={
                "api_key": api_key,
                "query": query,
                "max_results": max_results,
                "include_answer": True,
                "search_depth": "basic",
            },
            timeout=10,
        )
        if r.status_code != 200:
            return None
        data = r.json()
        delovi = []
        if data.get("answer"):
            delovi.append(f"**Sažetak:** {data['answer']}\n")
        for res in data.get("results", []):
            delovi.append(
                f"**{res.get('title', '')}**\n"
                f"  URL: {res.get('url', '')}\n"
                f"  {res.get('content', '')[:400]}"
            )
        return "\n\n".join(delovi) if delovi else None
    except Exception:
        return None


# ============================================================
# POMOĆNE FUNKCIJE
# ============================================================
def sredi_upit(t):
    if not t:
        return ""
    zamene_cir = {
        'а':'a','б':'b','в':'v','г':'g','д':'d','е':'e','ж':'ž','з':'z','и':'i',
        'ј':'j','к':'k','л':'l','м':'m','н':'n','о':'o','п':'p','р':'r','с':'s',
        'т':'t','у':'u','ф':'f','х':'h','ц':'c','ч':'č','ш':'š','ć':'ć','ђ':'đ',
    }
    out1 = []
    for c in str(t).lower():
        out1.append(zamene_cir.get(c, c))
    s = "".join(out1)
    zamene_dij = {'č':'c', 'ć':'c', 'š':'s', 'ž':'z', 'đ':'d'}
    return "".join([zamene_dij.get(c, c) for c in s])

def je_pitanje_za_eksterno(upit):
    u = sredi_upit(upit)
    ekstern_ključne = [
        "ko je ministar", "ko je predsednik", "ko je direktor pd",
        "ko je osnivac", "ko je osnovao", "ko je izumeo", "ko je napravio",
        "sta je", "sta su", "sta znaci", "sta predstavlja",
        "koji je", "koja je", "koje je",
        "kako se zove", "gde se nalazi", "kada je",
        "koliko kosta", "koliko je",
        "pd srbijasume", "javno preduzece", "preduzece za gazdovanje",
        "o kompaniji", "o firmi", "istorija", "kako posluje",
        "sediste", "kontakt", "veb sajt", "web sajt", "sajt",
    ]
    return any(kw in u for kw in ekstern_ključne)

def je_pitanje_o_kompaniji(upit):
    u = sredi_upit(upit)
    kompanija_ključne = [
        "pd srbijasume", "javno preduzece", "preduzece", "firma",
        "kompanija", "organizacija", "istorija", "delatnost",
        "sediste", "veb sajt", "web sajt",
    ]
    return any(kw in u for kw in kompanija_ključne)


def detektuj_tip(upit):
    u = sredi_upit(upit)
    if any(kw in u for kw in ["opiši sliku", "šta je na slici", "šta prikazuje", "opis slike", "analiziraj sliku"]):
        return "vizuelna_analiza"
    m = re.search(r'(?:clan|clana|clanom|clanu|clane|cl\.|cln\.|члан|члана|члану|чланом|чл\.)\s*(\d+)', u, re.IGNORECASE)
    if m:
        return f"clan_{m.group(1)}"
    if "direktor" in u or "директор" in u:
        return "direktor"
    words = u.split()
    if len(words) <= 3 and not any(kw in u for kw in ["stampac", "toner", "oprema", "dijagram", "mapa", "karta"]):
        return "osoba_ime"
    if "zaposlen" in u:
        indikatori = ["svi", "lista", "spisak", "koji su", "ko je sve", "navedi",
                      "ko radi", "kadrov", "imena", "ljudi", "ko je", "tko je"]
        if any(ind in u for ind in indikatori):
            return "lista_zaposlenih"
    if "toner" in u or "kertrid" in u:
        return "oprema"
    if any(kw in u for kw in ["stampac", "stampaci", "printer", "pisač", "oprema", "racunar"]):
        return "oprema"
    if any(kw in u for kw in ["dijagram", "mapa", "karta", "ruza vetrova", "vetrova",
                              "grafikon", "sema", "shema", "tabela", "skica",
                              "crtez", "prikaz", "pokaz", "slika", "fotografija"]):
        return "dijagram"
    return "standard"


def analyze_image_with_vision(image_url, question):
    api_token = os.environ.get("HF_API_TOKEN")
    if not api_token:
        return None
    API_URL = "https://api-inference.huggingface.co/models/Qwen/Qwen-VL-Chat"
    headers = {"Authorization": f"Bearer {api_token}"}
    try:
        response = requests.get(image_url, timeout=10)
        img = Image.open(io.BytesIO(response.content))
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode()
        payload = {
            "inputs": {
                "image": img_base64,
                "text": question,
            }
        }
        resp = requests.post(API_URL, headers=headers, json=payload, timeout=30)
        if resp.status_code == 200:
            return resp.json().get("generated_text", "Nema odgovora.")
        else:
            return None
    except Exception:
        return None


def get_point_count_by_tip():
    all_points = scroll_all()
    counts = {}
    for p in all_points:
        tip = p.payload.get("tip", "unknown")
        counts[tip] = counts.get(tip, 0) + 1
    return counts


# ============================================================
# STREAMLIT UI
# ============================================================
st.set_page_config(page_title="🌲 Biro asistent", page_icon="🌲", layout="wide")

col_l, col_c, col_r = st.columns([1, 3, 1])
with col_c:
    st.image(LOGO_URL, width=110)
    st.markdown(
        "<h2 style='text-align: center; color: #1b4332; margin: 0;'>🌲 Biro asistent</h2>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='text-align: center; color: #52796f; margin: 0;'>"
        "PD „Srbijašume” • Biro za planiranje</p>",
        unsafe_allow_html=True,
    )

st.markdown("---")

st.markdown("##### 💡 Brza pitanja:")
cols = st.columns(6)
QUICK = [
    "Ko je direktor Biroa?",
    "Ko su zaposleni u Birou?",
    "Koji su štampači u Birou?",
    "Koji toner se koristi?",
    "Prikaži dijagram ruže vetrova",
    "Član 14",
]
for i, label in enumerate(QUICK):
    with cols[i]:
        if st.button(label, use_container_width=True, key=f"q{i}"):
            st.session_state.pending = label

st.markdown("---")

with st.sidebar:
    st.image(LOGO_URL, width=80)
    st.markdown("### 🌲 Biro")
    st.caption(f"Kolekcija: {COLLECTION_NAME}")
    st.markdown("---")
    
    with st.expander("🔍 Debug info (broj tačaka po tipu)"):
        try:
            counts = get_point_count_by_tip()
            for tip, count in sorted(counts.items()):
                st.write(f"• **{tip}**: {count}")
        except Exception as e:
            st.error(f"Greška pri debug-u: {e}")
    
    st.markdown("---")
    if st.button("🧹 Obriši razgovor", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    if st.button("🔄 Osveži keš", use_container_width=True):
        st.cache_data.clear()
        st.session_state.cache_msg = "Keš podataka obrisan"
        st.rerun()
    if st.session_state.get("cache_msg"):
        st.success(f"✅ {st.session_state.cache_msg}")
        del st.session_state.cache_msg

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("images"):
            img_cols = st.columns(min(3, len(msg["images"])))
            for idx, (url, cap) in enumerate(msg["images"]):
                with img_cols[idx % 3]:
                    st.image(url, caption=cap, width=250)

user_input = st.chat_input("Postavite pitanje...")
if "pending" in st.session_state:
    user_input = st.session_state.pending
    del st.session_state.pending

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Tražim u bazi..."):
            try:
                tip = detektuj_tip(user_input)
                slike = []
                meta = ""

                if tip == "direktor":
                    odgovor, slike = handle_direktor()
                elif tip == "lista_zaposlenih":
                    odgovor, slike = handle_lista_zaposlenih()
                elif tip == "osoba_ime":
                    odgovor, slike = handle_osoba_po_imenu(user_input)
                elif tip == "oprema":
                    odgovor, slike = handle_oprema_specificno(user_input)
                elif tip == "dijagram":
                    odgovor, slike = handle_dijagram(user_input)
                elif tip.startswith("clan_"):
                    broj = tip.split("_")[1]
                    odgovor, slike = handle_clan(broj)
                elif tip == "vizuelna_analiza":
                    points = scroll_tip("dijagram")
                    if not points:
                        points = scroll_tip("fotografija_profil")
                    if points:
                        url = points[0].payload.get("slika_url", "") or points[0].payload.get("Link", "")
                        if url:
                            analiza = analyze_image_with_vision(url, user_input)
                            if analiza:
                                odgovor = f"**Analiza slike:**\n{analiza}"
                            else:
                                odgovor = "⚠️ Vizuelna analiza nije dostupna (nema HF tokena ili slika nije dostupna)."
                        else:
                            odgovor = "⚠️ Pronađena slika nema URL."
                    else:
                        odgovor = "⚠️ Nisam pronašao odgovarajuću sliku za analizu."
                else:
                    kontekst, br_k, slike, err = do_rag(user_input)
                    if err:
                        st.error(err)
                        odgovor = err
                    else:
                        ext_info = ""
                        koristio_ext = False
                        if je_pitanje_za_eksterno(user_input):
                            ext = external_search(user_input)
                            if ext:
                                ext_info = f"\n\n=== SPOLJNI IZVORI ===\n{ext}"
                                koristio_ext = True
                        if je_pitanje_o_kompaniji(user_input) or koristio_ext:
                            slike = []
                        messages = [
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": f"KONTEKST IZ BAZE:\n{kontekst}{ext_info}\n\nPitanje: {user_input}"},
                        ]
                        odgovor = ask_llm(messages)
                        meta = f"\n\n<sub>📊 Kandidati: {br_k}"
                        if koristio_ext:
                            meta += " | 🌐 Eksterno"
                        meta += "</sub>"

                if slike:
                    st.markdown("---")
                    img_cols = st.columns(min(3, len(slike)))
                    for idx, (url, cap) in enumerate(slike):
                        with img_cols[idx % 3]:
                            st.image(url, caption=cap, width=250)
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
                    "content": f"⚠️ Greška: {e}", "images": []
                })