"""
biro_app_v7.py v1.1 — Popravljena verzija
- Fix duplicate response
- Bolji routing
- Filteri po tip polju
- Robustni handleri
"""

import os
import sys
import re
import time
import requests
import uuid
from typing import List, Optional

import streamlit as st
from qdrant_client import QdrantClient
from qdrant_client import models

from langchain_core.documents import Document
from langchain_groq import ChatGroq

from biro_chain import BiroEmbeddings, get_qdrant_client, ensure_collections


# ============================================================
# CONFIG
# ============================================================
COLLECTION_PREFIX = os.environ.get("COLLECTION_PREFIX", "biro_v2")
TEXT_COLLECTION = f"{COLLECTION_PREFIX}_text"
IMAGE_COLLECTION = f"{COLLECTION_PREFIX}_images"

LOGO_URL = "https://pub-49fb3cc788a74e0a9edbac7e11305b94.r2.dev/logo_biro.png"

SYSTEM_PROMPT = """Ti si asistent za PD Srbijašume, Biro za planiranje.
Odgovaraj na srpskom, kratko i jasno. Koristi SAMO dati kontekst."""

QUICK_PROMPTS = [
    "Ко је директор Бироа?",
    "Који су запослени у Бироу?",
    "Који су штампачи у Бироу?",
    "Који тонeri се користе?",
    "Покажи дијаграм руже ветрова",
    "Члан 14",
]


# ============================================================
# CACHE
# ============================================================
@st.cache_resource
def get_embeddings():
    return BiroEmbeddings()


@st.cache_resource
def get_qdrant():
    q = get_qdrant_client()
    # Osiguraj da kolekcije i indeksi postoje (za stare i nove)
    try:
        ensure_collections(q, COLLECTION_PREFIX, text_dim=1024)
    except Exception as e:
        print(f"ensure_collections: {e}")
    # Osiguraj `tip` index i na STAROJ bazi (baza_cloud_v2_e5)
    try:
        for field in ("tip", "filename", "source", "Funkcija"):
            try:
                q.create_payload_index(
                    collection_name=LEGACY_COLLECTION,
                    field_name=field,
                    field_schema="keyword",
                )
            except Exception:
                pass
    except Exception as e:
        print(f"legacy index: {e}")
    return q


@st.cache_resource
def get_llm():
    return ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0.1,
        max_tokens=500,
        api_key=os.environ.get("GROQ_API_KEY"),
    )


# ============================================================
# UTILITY
# ============================================================
def sredi_upit(text: str) -> str:
    cyr_to_lat = {
        "а": "a", "б": "b", "в": "v", "г": "g", "д": "d", "ђ": "đ",
        "е": "e", "ж": "ž", "з": "z", "и": "i", "ј": "j", "к": "k",
        "л": "l", "љ": "lj", "м": "m", "н": "n", "њ": "nj", "о": "o",
        "п": "p", "р": "r", "с": "s", "т": "t", "ћ": "ć", "у": "u",
        "ф": "f", "х": "h", "ц": "c", "ч": "č", "џ": "dž", "ш": "š",
    }
    text_lower = text.lower()
    for cyr, lat in cyr_to_lat.items():
        text_lower = text_lower.replace(cyr, lat)
    return (text_lower.replace("č", "c").replace("ć", "c")
                  .replace("š", "s").replace("ž", "z").replace("đ", "dj"))


# ============================================================
# RETRIEVER (optimizovan)
# ============================================================
LEGACY_COLLECTION = "baza_cloud_v2_e5"  # stara baza, 768d, klasifikovana


def retrieve(query: str, k: int = 10, tip_filter: str = None, source: str = None) -> List[dict]:
    """Semantic search u NOVOJ bazi (biro_v2_text, 1024d)."""
    embeddings = get_embeddings()
    qdrant = get_qdrant()

    q_vec = embeddings.embed_query(query)

    must = []
    if tip_filter:
        must.append(models.FieldCondition(key="tip", match=models.MatchValue(value=tip_filter)))
    if source:
        must.append(models.FieldCondition(key="source", match=models.MatchValue(value=source)))
    search_filter = models.Filter(must=must) if must else None

    try:
        results = qdrant.query_points(
            collection_name=TEXT_COLLECTION,
            query=q_vec,
            limit=k,
            query_filter=search_filter,
        )
        docs = []
        for point in results.points:
            payload = point.payload or {}
            docs.append({
                "text": payload.get("text", ""),
                "filename": payload.get("filename", ""),
                "tip": payload.get("tip", ""),
                "source": payload.get("source", ""),
                "Funkcija": payload.get("Funkcija", ""),
                "Link": payload.get("Link", ""),
                "score": point.score,
            })
        return docs
    except Exception as e:
        st.error(f"Greška pri pretrazi: {e}")
        return []


def retrieve_legacy(tip: str = None, k: int = 100) -> List[dict]:
    """Scroll sa filterom u STAROJ bazi (baza_cloud_v2_e5, 768d).
    Koristi se za specifične upite gde `tip` polje već postoji."""
    qdrant = get_qdrant()

    must = []
    if tip:
        must.append(models.FieldCondition(key="tip", match=models.MatchValue(value=tip)))
    scroll_filter = models.Filter(must=must) if must else None

    try:
        results, _ = qdrant.scroll(
            collection_name=LEGACY_COLLECTION,
            limit=k,
            scroll_filter=scroll_filter,
            with_payload=True,
            with_vectors=False,
        )
        docs = []
        for r in results:
            payload = r.payload or {}
            # Legacy baza koristi `tekst` (ne `text`), `izvor` (ne `filename`)
            tekst = payload.get("tekst", "") or payload.get("text", "")
            objekat = payload.get("Objekat", "") or payload.get("objekat", "")
            link = payload.get("Link", "") or payload.get("link", "")
            docs.append({
                "text": tekst,
                "objekat": objekat,
                "izvor": payload.get("izvor", ""),
                "tip": payload.get("tip", ""),
                "Link": link,
                "Funkcija": payload.get("Funkcija", ""),
            })
        return docs
    except Exception as e:
        st.error(f"Greška u legacy pretraga: {e}")
        return []


# ============================================================
# EXTERNAL SEARCH
# ============================================================
def external_search(query: str) -> Optional[str]:
    api_key = os.environ.get("TAVILY_API_KEY", "")
    if not api_key:
        return None
    try:
        r = requests.post(
            "https://api.tavily.com/search",
            json={
                "api_key": api_key,
                "query": query,
                "max_results": 5,
                "include_answer": False,
                "search_depth": "basic",
            },
            timeout=10,
        )
        if r.status_code != 200:
            return None
        data = r.json()
        delovi = []
        for res in data.get("results", []):
            delovi.append(
                f"📄 **{res.get('title', '')}**\n"
                f"   🔗 {res.get('url', '')}\n"
                f"   {res.get('content', '')[:300]}"
            )
        return "\n\n".join(delovi) if delovi else "Nema rezultata."
    except Exception:
        return None


# ============================================================
# ROUTING
# ============================================================
def detektuj_tip(upit: str) -> str:
    u = sredi_upit(upit)

    # Pravni član
    m = re.search(r'(?:clan|clana|clanom|clanu|clane|cl\.|cln\.)\s*(\d+)', u)
    if m:
        return f"clan_{m.group(1)}"

    # Eksterno
    if re.search(r'\b(minist|predsed|premijer|vladar|kralj|guverne|ambasad)', u):
        if "direktor" not in u and "biro" not in u:
            return "eksterno"

    ekstern_opste = ["sta je", "sta su", "sta znaci", "gde se", "gde je",
                     "koliko", "kada je", "kako da", "zasto", "koja je adresa", "adresa"]
    if any(kw in u for kw in ekstern_opste):
        if "zaposlen" not in u and "biro" not in u:
            return "eksterno"

    if "direktor" in u and "zamenik" not in u:
        return "direktor"

    if any(kw in u for kw in ["ima li", "pronadi", "nadji", "ko je ", "kako se zove"]):
        if "zaposlen" not in u and "lista" not in u:
            return "osoba"

    if "zaposlen" in u and any(kw in u for kw in ["svi", "lista", "spisak", "koji su", "imena"]):
        return "lista"

    if "toner" in u or "kertrid" in u or any(kw in u for kw in ["stampac", "oprema", "racunar"]):
        return "oprema"

    if any(kw in u for kw in ["dijagram", "mapa", "karta", "ruza vetrova", "vetrova", "shema"]):
        return "dijagram"

    return "standard"


# ============================================================
# HANDLERI
# ============================================================
def _parse_ime(tekst: str):
    """Parsira 'Fotografija Ime Prezime, pozicija, ...' iz teksta."""
    m = re.search(
        r'[Ff]otografij[ae]\s+([A-ZČĆŠĐŽ][a-zčćšđžA-Z]+\s+[A-ZČĆŠĐŽ][a-zčćšđžA-Z]+)',
        tekst
    )
    if m:
        return m.group(1).strip()
    return None


def _parse_pozicija(tekst: str):
    """Parsira poziciju iz 'Fotografija Ime, pozicija, PD Srbijašume, ...'."""
    m = re.search(
        r'[Ff]otografij[ae]\s+[A-ZČĆŠĐŽ][a-zčćšđžA-Z]+\s+[A-ZČĆŠĐŽ][a-zčćšđžA-Z]+,\s*([^,]+),',
        tekst
    )
    if m:
        return m.group(1).strip()
    return None


def handle_direktor():
    """Traži direktora u legacy bazi."""
    docs = retrieve_legacy(tip="fotografija_profil", k=200)

    direktori = []
    zamenici = []
    seen = set()
    slike = []

    for d in docs:
        tekst = d.get("text", "")
        ime = _parse_ime(tekst)
        if ime and ime not in seen and len(ime) > 5:
            seen.add(ime)
            lower = tekst.lower()
            link = d.get("Link", "")
            entry = {"ime": ime, "link": link, "pozicija": _parse_pozicija(tekst) or ""}
            if "zamenik" in lower and "direktor" in lower:
                zamenici.append(entry)
            elif "direktor" in lower:
                direktori.append(entry)

    output = ""
    if direktori:
        output += f"**Direktor ({len(direktori)}):**\n"
        for d in direktori:
            output += f"- **{d['ime']}** — {d['pozicija']}\n"
            if d["link"]:
                slike.append((d["link"], d["ime"]))
    if zamenici:
        output += f"\n**Zamenici direktora ({len(zamenici)}):**\n"
        for z in zamenici:
            output += f"- **{z['ime']}** — {z['pozicija']}\n"
            if z["link"]:
                slike.append((z["link"], z["ime"]))

    return output or "Nema rezultata za direktora u Birou.", slike


def handle_lista_zaposlenih():
    """Svi zaposleni sa fotografijama."""
    docs = retrieve_legacy(tip="fotografija_profil", k=200)

    zaposleni = []
    seen = set()
    slike = []

    for d in docs:
        tekst = d.get("text", "")
        ime = _parse_ime(tekst)
        if ime and ime not in seen and len(ime) > 5:
            seen.add(ime)
            link = d.get("Link", "")
            zaposleni.append({
                "ime": ime,
                "link": link,
                "pozicija": _parse_pozicija(tekst) or "",
            })

    if not zaposleni:
        return "⚠️ Nisu pronađeni zaposleni u bazi.", []

    zaposleni.sort(key=lambda x: x["ime"])

    output = f"**Zaposleni u Birou ({len(zaposleni)}):**\n"
    for z in zaposleni:
        output += f"- {z['ime']} — {z['pozicija']}\n"
        if z["link"]:
            slike.append((z["link"], z["ime"]))
    return output, slike


def handle_oprema(upit: str):
    """Oprema - koristi legacy bazu (gde je `tip=oprema`)."""
    u = upit.lower()
    samo_toner = any(kw in u for kw in ["toner", "kertrid", "kertridž", "cartridge"])
    samo_stampac = any(kw in u for kw in ["stampac", "štampac", "štampač", "printer"])
    samo_racunar = any(kw in u for kw in ["racun", "račun", "kompjut", "laptop", "monitor", "skener"])

    svi_docs = []
    seen_ids = set()

    # Legacy baza (gde je oprema klasifikovana)
    for d in retrieve_legacy(tip="oprema", k=500):
        key = d.get("text", "")[:100]
        if key not in seen_ids:
            seen_ids.add(key)
            svi_docs.append(d)

    # Ako nema dovoljno, probaj i u v2
    if len(svi_docs) < 5:
        for d in retrieve("oprema štampač toner računar", k=30):
            key = d.get("text", "")[:100]
            if key not in seen_ids:
                seen_ids.add(key)
                svi_docs.append(d)

    # Bolji regex: Kyocera SAMO ako ima model
    printer_pat = re.compile(
        r'\b(?:'
        r'Kyocera\s+(?:TASKalfa|FS-\d+|ECOSYS\s+[\w-]+|M\d{4}|P\d{4})'
        r'|HP\s+(?:LaserJet|OfficeJet|PageWide|Designjet)\s+[\w]+'
        r'|Canon\s+(?:imageRUNNER|PIXMA|TX-\d+)'
        r')\b', re.IGNORECASE
    )
    toner_pat = re.compile(
        r'\b(?:'
        r'TK-\d+\w*'
        r'|HP\s+(?:[CP]\d+\w*|CE\d+\w*|C4\d{3}\w*)'
        r'|Canon\s+(?:PFI-\d+\w*|CL-\d+\w*|PGI-\d+\w*)'
        r')\b', re.IGNORECASE
    )

    stampaci, toneri = set(), set()
    for d in svi_docs:
        text = d.get("text", "")
        for m in printer_pat.findall(text):
            stampaci.add(m.strip())
        for m in toner_pat.findall(text):
            toneri.add(m.strip())

    # Ako je samo jedna kategorija, filtriraj
    if samo_toner and not samo_stampac:
        stampaci = set()
    if samo_stampac and not samo_toner:
        toneri = set()

    output = ""
    if samo_toner:
        if toneri:
            output = f"**Toneri ({len(toneri)}):**\n"
            output += "\n".join(f"- {t}" for t in sorted(toneri))
        else:
            output = "⚠️ Nema tonera u bazi."
    elif samo_stampac:
        if stampaci:
            output = f"**Štampači ({len(stampaci)}):**\n"
            output += "\n".join(f"- {s}" for s in sorted(stampaci))
        else:
            output = "⚠️ Nema štampača u bazi."
    else:
        if stampaci:
            output += f"**Štampači ({len(stampaci)}):**\n"
            output += "\n".join(f"- {s}" for s in sorted(stampaci)) + "\n\n"
        if toneri:
            output += f"**Toneri ({len(toneri)}):**\n"
            output += "\n".join(f"- {t}" for t in sorted(toneri))
        if not output:
            output = "⚠️ Nema specifične opreme u bazi (probaj drugi upit)."

    return output, []


def handle_dijagram(upit: str):
    """Dijagram - koristi legacy bazu."""
    u = upit.lower()
    trazim_wind_rose = any(kw in u for kw in ["vetr", "rose", "ruza", "wind"])

    svi_docs = []
    seen_ids = set()

    # 1. Legacy baza (svi dijagrami)
    for d in retrieve_legacy(tip="dijagram", k=500):
        key = d.get("text", "")[:100]
        if key not in seen_ids:
            seen_ids.add(key)
            svi_docs.append(d)

    # 2. Embedding search u v2
    for d in retrieve("dijagram šema grafikon", k=15):
        key = d.get("text", "")[:100]
        if key not in seen_ids:
            seen_ids.add(key)
            svi_docs.append(d)

    # Filtriraj po ključnim rečima u tekstu ILI filename
    relevantni = []
    for d in svi_docs:
        text = d.get("text", "").lower()
        izvor = d.get("izvor", "").lower()
        haystack = text + " " + izvor
        if trazim_wind_rose:
            if any(kw in haystack for kw in ["wind", "rose", "ruza", "vetrova", "vetar",
                                              "klimatski", "klim"]):
                relevantni.append(d)
        else:
            if any(kw in haystack for kw in ["dijagram", "grafikon", "shema", "mapa",
                                              "karta", "skica"]):
                relevantni.append(d)

    if not relevantni:
        # Ako ništa specifično, prikaži prvih 5 dijagrama
        relevantni = svi_docs[:5]
        info = f"⚠️ Nema specifičnog '{upit}', ali evo {len(relevantni)} dijagrama iz baze:\n\n"
    else:
        info = f"**Pronađeno {len(relevantni)} dijagrama:**\n"

    output = info
    for i, d in enumerate(relevantni[:8]):
        text = d.get("text", "")[:250]
        izvor = d.get("izvor", "")
        output += f"\n[{i+1}] {izvor}\n{text}...\n"

    return output, []


def handle_clan(broj: str):
    """Pravni član - koristi legacy bazu (`tip=pravni_akt`)."""
    # Fleksibilniji pattern: hvata razne oblike
    clan_pat = re.compile(
        rf'(?:[Čč]lan|[Čč]l\.?|[Cc]lan|[Cc]l\.?)\s*\.?\s*{re.escape(broj)}\b',
        re.IGNORECASE
    )
    # Kraj člana: sledeći član ili "\n\n"
    kraj_pat = re.compile(
        rf'(?:[Čč]lan|[Čč]l\.?|[Cc]lan|[Cc]l\.?)\s*\.?\s*\d+\b',
        re.IGNORECASE
    )

    pogodci = []
    seen = set()

    # Legacy baza sa filterom
    for d in retrieve_legacy(tip="pravni_akt", k=2000):
        text = d.get("text", "")
        for m in clan_pat.finditer(text):
            start = m.start()
            # Nađi kraj (sledeći član)
            end_m = kraj_pat.search(text, m.end())
            end = end_m.start() if end_m else min(start + 4000, len(text))
            clan_tekst = text[start:end].strip()
            if 30 < len(clan_tekst) < 5000:
                k = sredi_upit(clan_tekst[:200])
                if k not in seen:
                    seen.add(k)
                    pogodci.append({"tekst": clan_tekst, "izvor": d.get("izvor", "")})

    # Fallback: v2
    if not pogodci:
        for d in retrieve(f"clan {broj} ugovor zakon pravilnik", k=50):
            text = d.get("text", "")
            for m in clan_pat.finditer(text):
                start = m.start()
                end_m = kraj_pat.search(text, m.end())
                end = end_m.start() if end_m else min(start + 4000, len(text))
                clan_tekst = text[start:end].strip()
                if 30 < len(clan_tekst) < 5000:
                    k = sredi_upit(clan_tekst[:200])
                    if k not in seen:
                        seen.add(k)
                        pogodci.append({"tekst": clan_tekst, "izvor": d.get("filename", "")})

    if not pogodci:
        return f"⚠️ Član {broj} nije pronađen.", []

    pogodci.sort(key=lambda x: -len(x["tekst"]))
    p = pogodci[0]
    return f"**Члан {broj}** (izvor: {p['izvor']}):\n\n{p['tekst']}", []


def handle_osoba_po_imenu(upit: str):
    """Pretraga po imenu — kombinacija obe baze."""
    svi_docs = []
    seen = set()

    for d in retrieve_legacy(tip="fotografija_profil", k=200):
        key = d.get("text", "")[:100]
        if key not in seen:
            seen.add(key)
            svi_docs.append(d)
    for d in retrieve(upit, k=15):
        key = d.get("text", "")[:100]
        if key not in seen:
            seen.add(key)
            svi_docs.append(d)

    # Filtriraj po imenu u tekstu
    sredi = sredi_upit(upit)
    svi_docs = [d for d in svi_docs if any(
        kw in d.get("text", "").lower() for kw in upit.lower().split() if len(kw) > 3
    )]

    if not svi_docs:
        return "Nema rezultata.", []

    output = f"Pronađeno {len(svi_docs)} rezultata:\n\n"
    for i, d in enumerate(svi_docs[:5]):
        text = d.get("text", "")[:200]
        izvor = d.get("izvor", "")
        output += f"**[{i+1}]** {izvor}\n{text}...\n\n"
    return output, []


def handle_standard(upit: str):
    """Standard RAG."""
    docs = retrieve(upit, k=10)
    if not docs:
        return "Nema rezultata u bazi.", 0, []

    kontekst = "\n\n---\n\n".join(d["text"] for d in docs[:5])

    ext_info = ""
    if detektuj_tip(upit) == "eksterno":
        ext = external_search(upit)
        if ext:
            ext_info = f"\n\n=== SPOLJNI IZVORI ===\n{ext}"

    try:
        llm = get_llm()
        response = llm.invoke([
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"KONTEKST:\n{kontekst}{ext_info}\n\nPitanje: {upit}"},
        ])
        odgovor = response.content
    except Exception as e:
        odgovor = f"⚠️ LLM greška: {e}"

    return odgovor, len(docs), []


# ============================================================
# STREAMLIT UI
# ============================================================
st.set_page_config(page_title="Биро асистент v7", page_icon="🌲", layout="wide")

col_l, col_c, col_r = st.columns([1, 3, 1])
with col_c:
    st.image(LOGO_URL, width=110)
    st.markdown(
        "<h2 style='text-align: center; color: #1b4332; margin: 0;'>🌲 Биро асистент <small>v7</small></h2>",
        unsafe_allow_html=True,
    )

st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("⚙️ Podešavanja")
    st.write(f"**Kolekcija:** `{TEXT_COLLECTION}`")

    if st.button("🔍 Diagnostika"):
        with st.spinner("Provera..."):
            try:
                qd = get_qdrant()
                cols = [c.name for c in qd.get_collections().collections]
                st.write("**Kolekcije u Qdrant:**")
                for c in cols:
                    info = qd.get_collection(c)
                    count = (getattr(info, "points_count", None) or
                             getattr(info, "vectors_count", None) or 0)
                    st.write(f"- `{c}`: {count} tačaka")

                # Test embedding
                emb = get_embeddings()
                test_vec = emb.embed_query("test")
                st.write(f"**Embedding dim:** {len(test_vec)}")

                # Test v2 search
                results = qd.query_points(
                    collection_name=TEXT_COLLECTION,
                    query=test_vec,
                    limit=3,
                )
                st.write(f"**v2 Test pretraga:** {len(results.points)} rezultata")
                for p in results.points:
                    st.write(f"  - {p.payload.get('filename', '?')} (tip={p.payload.get('tip', '?')})")

                # ===== LEGACY: broj zapisa po tipu =====
                st.write("---")
                st.write("**Legacy baza — `tip` vrednosti:**")
                all_results, _ = qd.scroll(
                    collection_name=LEGACY_COLLECTION,
                    limit=5000,
                    with_payload=True,
                    with_vectors=False,
                )
                tip_count = {}
                sample_po_tipu = {}
                for r in all_results:
                    t = (r.payload or {}).get("tip", "<nEMA>")
                    tip_count[t] = tip_count.get(t, 0) + 1
                    if t not in sample_po_tipu:
                        sample_po_tipu[t] = (r.payload or {}).get("filename", "?")
                for t, n in sorted(tip_count.items(), key=lambda x: -x[1]):
                    st.write(f"  - `{t}`: {n} (primer: {sample_po_tipu[t]})")

                # Test legacy filter
                st.write("---")
                st.write("**Legacy filter test `tip=fotografija_profil`:**")
                try:
                    f_results, _ = qd.scroll(
                        collection_name=LEGACY_COLLECTION,
                        limit=5,
                        scroll_filter=models.Filter(
                            must=[models.FieldCondition(
                                key="tip",
                                match=models.MatchValue(value="fotografija_profil")
                            )]
                        ),
                        with_payload=True,
                        with_vectors=False,
                    )
                    st.write(f"Pronađeno: {len(f_results)} (prikazujem 5)")
                    for i, r in enumerate(f_results[:5]):
                        payload = r.payload or {}
                        st.write(f"**Zapis {i+1}** — sva polja:")
                        for k, v in payload.items():
                            v_str = str(v)[:200]
                            st.write(f"  - `{k}`: {v_str}")
                except Exception as e:
                    st.error(f"Filter test: {e}")

            except Exception as e:
                import traceback
                st.error(f"Greška: {e}")
                st.code(traceback.format_exc())

    if st.button("🔄 Osveži keš"):
        st.cache_resource.clear()
        st.rerun()

    if st.button("🗑️ Obriši razgovor"):
        st.session_state.messages = []
        st.rerun()

# Quick prompts
st.markdown("##### 💡 Brza pitanja:")
cols = st.columns(6)
for i, label in enumerate(QUICK_PROMPTS):
    with cols[i]:
        if st.button(label, use_container_width=True, key=f"q{i}"):
            st.session_state.pending = label

st.markdown("---")

# Chat
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending" not in st.session_state:
    st.session_state.pending = None

# Prikaz CELOKUPNE istorije
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg.get("images"):
            for url, cap in msg.get("images", []):
                st.image(url, caption=cap, width=250)

# Pending pitanje
if st.session_state.pending:
    user_input = st.session_state.pending
    st.session_state.pending = None
else:
    user_input = st.chat_input("Postavi pitanje...")

if user_input:
    # Dodaj user poruku u istoriju
    st.session_state.messages.append({"role": "user", "content": user_input, "images": []})
    with st.chat_message("user"):
        st.write(user_input)

    # Process
    with st.chat_message("assistant"):
        with st.spinner("..."):
            tip = detektuj_tip(user_input)
            slike = []
            meta = ""

            try:
                if tip == "direktor":
                    odgovor, slike = handle_direktor()
                elif tip == "lista":
                    odgovor, slike = handle_lista_zaposlenih()
                elif tip == "osoba":
                    odgovor, slike = handle_osoba_po_imenu(user_input)
                elif tip == "oprema":
                    odgovor, slike = handle_oprema(user_input)
                elif tip == "dijagram":
                    odgovor, slike = handle_dijagram(user_input)
                elif tip.startswith("clan_"):
                    broj = tip.split("_")[1]
                    odgovor, slike = handle_clan(broj)
                elif tip == "eksterno":
                    ext = external_search(user_input)
                    if ext:
                        odgovor = f"🌐 **Spoljni izvori:**\n\n{ext}"
                        meta = "\n\n<sub>🌐 Eksterni search</sub>"
                    else:
                        odgovor = "⚠️ Nemam pristup eksternim izvorima."
                    slike = []
                else:
                    odgovor, br_k, slike = handle_standard(user_input)
                    if br_k:
                        meta = f"\n\n<sub>📊 Kandidati: {br_k}</sub>"
            except Exception as e:
                odgovor = f"⚠️ Greška: {e}"
                slike = []

            # Prikaz SADA
            if slike:
                st.markdown("---")
                for url, cap in slike:
                    st.image(url, caption=cap, width=250)
                st.markdown("---")

            st.markdown(odgovor + meta)

    # Dodaj u istoriju
    st.session_state.messages.append({
        "role": "assistant",
        "content": odgovor + meta,
        "images": slike,
    })
