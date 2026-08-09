"""
app.py — LangChain Streamlit aplikacija
=================================================
Kompletna RAG aplikacija sa:
- LangChain za orkestraciju
- bge-m3 embeddings (multilingual, ćirilica)
- CLIP za slike
- Groq LLM (free, 30 req/min)
- Tavily za eksterne pretrage
- Multi-Vector retriever (text + image)

Env varijable:
    QDRANT_URL, QDRANT_API_KEY
    GROQ_API_KEY
    TAVILY_API_KEY (opciono)
    COLLECTION_PREFIX (default: biro_v2)

Pokretanje:
    streamlit run app.py
"""

import os
COLLECTION_PREFIX = os.environ.get("COLLECTION_PREFIX", "biro_v2")
import sys
import re
import time
import base64
import requests
import hashlib
from pathlib import Path
from typing import List, Optional

import streamlit as st
from qdrant_client import QdrantClient
from qdrant_client import models

# LangChain
from langchain_core.documents import Document
from langchain_groq import ChatGroq

# Naš LangChain modul
from biro_chain import (
    BiroEmbeddings,
    get_qdrant_client,
)


# ============================================================
# CONFIG
# ============================================================
COLLECTION_PREFIX = os.environ.get("COLLECTION_PREFIX", "biro_v2")
TEXT_COLLECTION = f"{COLLECTION_PREFIX}_text"
IMAGE_COLLECTION = f"{COLLECTION_PREFIX}_images"

LOGO_URL = "https://pub-49fb3cc788a74e0a9edbac7e11305b94.r2.dev/logo_biro.png"

SYSTEM_PROMPT = """Ti si asistent za PD Srbijašume, Biro za planiranje.
Odgovaraj na srpskom, kratko i jasno. Koristi SAMO dati kontekst.
Ako nema dovoljno informacija, reci "Nemam dovoljno informacija"."""

QUICK_PROMPTS = [
    "Ко је директор Бироа?",
    "Који су запослени у Бироу?",
    "Који су штампачи у Бироу?",
    "Који тонeri се користе?",
    "Покажи дијаграм руже ветрова",
    "Члан 14",
]


# ============================================================
# CACHE: embeddings + Qdrant
# ============================================================
@st.cache_resource
def get_embeddings():
    """Keširamo embeddings (modeli su teški)."""
    return BiroEmbeddings()


@st.cache_resource
def get_qdrant():
    return get_qdrant_client()


@st.cache_resource
def get_llm():
    return ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0.1,
        max_tokens=500,
        api_key=os.environ.get("GROQ_API_KEY"),
    )


# ============================================================
# UTILITY: text cleaning
# ============================================================
def sredi_upit(text: str) -> str:
    """Ćirilica → latinicu + skidanje dijakritike za pretragu."""
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
    # Skini dijakritike
    text_lower = (text_lower.replace("č", "c").replace("ć", "c")
                  .replace("š", "s").replace("ž", "z").replace("đ", "dj"))
    return text_lower


# ============================================================
# RETRIEVER
# ============================================================
def retrieve(query: str, k: int = 10) -> List[Document]:
    """Pretraga Qdrant baze po tekstu (semantic search)."""
    embeddings = get_embeddings()
    qdrant = get_qdrant()

    # Embedding upita
    q_vec = embeddings.embed_query(query)

    try:
        results = qdrant.query_points(
            collection_name=TEXT_COLLECTION,
            query=q_vec,
            limit=k,
        )
        docs = []
        for point in results.points:
            payload = point.payload or {}
            text = payload.get("text", "")
            docs.append({
                "text": text,
                "filename": payload.get("filename", ""),
                "source": payload.get("source", ""),
                "chunk_index": payload.get("chunk_index", 0),
                "score": point.score,
            })
        return docs
    except Exception as e:
        st.error(f"Greška pri pretrazi: {e}")
        return []


def retrieve_images(query: str, k: int = 5) -> List[dict]:
    """Pretraga baze slika (image embeddings)."""
    embeddings = get_embeddings()
    qdrant = get_qdrant()

    # Za image embedding treba image, ne text. Za sada preskačemo.
    # Ako imaš tekst, koristi OCR na upit + CLIP.
    return []


# ============================================================
# EXTERNAL SEARCH (Tavily)
# ============================================================
def external_search(query: str, max_results: int = 5) -> Optional[str]:
    """Tavily pretraga — bez AI sažetka."""
    api_key = os.environ.get("TAVILY_API_KEY", "")
    if not api_key:
        return None
    try:
        r = requests.post(
            "https://api.tavily.com/search",
            json={
                "api_key": api_key,
                "query": query,
                "max_results": max_results,
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
    """Vraća tip zahteva: direktor, lista, oprema, dijagram, clan_N, eksterno, standard."""
    u = sredi_upit(upit)

    # Pravni član
    m = re.search(r'(?:clan|clana|clanom|clanu|clane|cl\.|cln\.)\s*(\d+)', u)
    if m:
        return f"clan_{m.group(1)}"

    # Eksterno (poznate ličnosti)
    if re.search(r'\b(minist|predsed|premijer|vladar|kralj|guverne|ambasad|potpredsed)', u):
        if "direktor" not in u and "biro" not in u:
            return "eksterno"

    # Opšte eksterno
    ekstern_opste = [
        "sta je", "sta su", "sta znaci", "gde se", "gde je",
        "koliko", "kada je", "kako da", "zasto",
        "koja je adresa", "adresa",
    ]
    if any(kw in u for kw in ekstern_opste):
        if "zaposlen" not in u and "biro" not in u:
            return "eksterno"

    # Direktor
    if "direktor" in u and "zamenik" not in u:
        return "direktor"

    # Osoba
    if any(kw in u for kw in ["ima li", "pronadi", "nadji", "ko je ", "kako se zove"]):
        if "zaposlen" not in u and "lista" not in u:
            return "osoba"

    # Lista zaposlenih
    if "zaposlen" in u and any(kw in u for kw in ["svi", "lista", "spisak", "koji su", "imena"]):
        return "lista"

    # Oprema
    if "toner" in u or "kertrid" in u or any(kw in u for kw in ["stampac", "oprema", "racunar"]):
        return "oprema"

    # Dijagram
    if any(kw in u for kw in ["dijagram", "mapa", "karta", "ruza vetrova", "vetrova", "shema"]):
        return "dijagram"

    return "standard"


# ============================================================
# HANDLERI
# ============================================================
def handle_direktor():
    """Pronađi direktora po 'Funkcija=direktor' polju."""
    qdrant = get_qdrant()
    results = []
    offset = None
    while True:
        records, next_offset = qdrant.scroll(
            collection_name=TEXT_COLLECTION,
            limit=100,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        for r in records:
            payload = r.payload or {}
            if payload.get("type") == "image" or "Fotografij" in (payload.get("text", "") or "")[:200]:
                text = payload.get("text", "")
                m = re.search(r'[Ff]otografij[ae]\s+([A-ZČĆŠĐŽ][a-zčćšđž]+\s+[A-ZČĆŠĐŽ][a-zčćšđž]+)', text)
                if m:
                    results.append({
                        "ime": m.group(1),
                        "funkcija": payload.get("metadata", {}).get("funkcija", ""),
                    })
        if next_offset is None or len(records) == 0:
            break
        offset = next_offset

    if not results:
        return "⚠️ Nisu pronađeni direktor/zamenici u photo zapisima.", []

    direktori = [r for r in results if "direktor" in r.get("funkcija", "").lower() and "zamenik" not in r.get("funkcija", "").lower()]
    zamenici = [r for r in results if "zamenik" in r.get("funkcija", "").lower()]

    output = ""
    if direktori:
        output += f"**Direktor:**\n" + "\n".join(f"- {r['ime']}" for r in direktori) + "\n\n"
    if zamenici:
        output += f"**Zamenici:**\n" + "\n".join(f"- {r['ime']}" for r in zamenici)

    return output or "Nema rezultata.", []


def handle_lista_zaposlenih():
    """Svi zaposleni iz foto zapisa."""
    qdrant = get_qdrant()
    zaposleni = []
    offset = None
    while True:
        records, next_offset = qdrant.scroll(
            collection_name=TEXT_COLLECTION,
            limit=200,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        for r in records:
            payload = r.payload or {}
            text = payload.get("text", "")
            m = re.search(r'[Ff]otografij[ae]\s+([A-ZČĆŠĐŽ][a-zčćšđž]+\s+[A-ZČĆŠĐŽ][a-zčćšđž]+)', text)
            if m:
                zaposleni.append(m.group(1))
        if next_offset is None or len(records) == 0:
            break
        offset = next_offset

    if not zaposleni:
        return "⚠️ Nisu pronađeni zaposleni u photo zapisima.", []

    zaposleni = sorted(set(zaposleni))
    output = f"**Zaposleni u Birou ({len(zaposleni)}):**\n"
    output += "\n".join(f"- {z}" for z in zaposleni)
    return output, []


def handle_oprema(upit: str):
    """Specifična oprema (štampači, toneri)."""
    u = sredi_upit(upit)
    docs = retrieve(upit, k=20)

    stampaci_patterns = re.compile(
        r'\b(?:Kyocera|HP\s+\w+|Canon\s+\w+|TASKalfa|ECOSYS|'
        r'FS-\d+|M\d{4}|P\d{4}|imageRUNNER|PIXMA)\b', re.IGNORECASE)

    toner_patterns = re.compile(
        r'\b(?:TK-\d+|HP\s+[CP]\d+|CE\d+|'
        r'Canon\s+(?:PFI|CL)-\d+)\b', re.IGNORECASE)

    stampaci, toneri = set(), set()
    for d in docs:
        text = d.get("text", "")
        for m in stampaci_patterns.findall(text):
            stampaci.add(m)
        for m in toner_patterns.findall(text):
            toneri.add(m)

    output = ""
    if stampaci:
        output += f"**Štampači ({len(stampaci)}):**\n"
        output += "\n".join(f"- {s}" for s in sorted(stampaci)) + "\n\n"
    if toneri:
        output += f"**Toneri ({len(toneri)}):**\n"
        output += "\n".join(f"- {t}" for t in sorted(toneri))

    return output or "⚠️ Nema opreme u bazi.", []


def handle_dijagram(upit: str):
    """Dijagram — skroluj slike sa vizuel_opis poljem."""
    qdrant = get_qdrant()
    dijagrami = []
    offset = None
    while True:
        records, next_offset = qdrant.scroll(
            collection_name=TEXT_COLLECTION,
            limit=200,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        for r in records:
            payload = r.payload or {}
            text = (payload.get("text", "") or "")[:200]
            if "Slika" in text or "fotografij" in text.lower():
                continue
            dijagrami.append({
                "text": payload.get("text", ""),
                "filename": payload.get("filename", ""),
                "vizuel_opis": payload.get("vizuel_opis", ""),
            })
        if next_offset is None or len(records) == 0:
            break
        offset = next_offset

    if not dijagrami:
        return "⚠️ Nema dijagrama u bazi.", []

    u = sredi_upit(upit)
    lokacije = ["crni vrh", "stig", "vranjaca", "donji pek", "beograd"]
    target = None
    for lok in lokacije:
        if lok in u:
            target = lok
            break

    if target:
        filtrirani = [d for d in dijagrami if target in (d.get("vizuel_opis", "") + d.get("filename", "")).lower()]
        if filtrirani:
            return f"**Pronađeno {len(filtrirani)} dijagrama za '{target}':**", []

    return f"**Pronađeno {len(dijagrami)} dijagrama (prvih 6):**", []


def handle_clan(broj: str):
    """Pravni član."""
    pattern = re.compile(
        rf'(?:clan|clana|clanom|clanu|clane|cl\.|cln\.)\s*{re.escape(broj)}\b(.*?)'
        rf'(?=(?:clan|clana|clanom|clanu|clane|cl\.|cln\.)\s*\d+\b|$)',
        re.DOTALL | re.IGNORECASE
    )

    docs = retrieve(f"clan {broj}", k=50)
    pogodci = []
    for d in docs:
        text = d.get("text", "")
        for m in pattern.finditer(text):
            clan_tekst = m.group(0).strip()
            if 30 < len(clan_tekst) < 4000:
                pogodci.append({
                    "tekst": clan_tekst,
                    "izvor": d.get("filename", ""),
                })

    if not pogodci:
        return f"⚠️ Član {broj} nije pronađen.", []

    # Dedupe
    seen = set()
    uniq = []
    for p in pogodci:
        k = sredi_upit(p["tekst"][:200])
        if k not in seen:
            seen.add(k)
            uniq.append(p)

    uniq.sort(key=lambda x: -len(x["tekst"]))
    p = uniq[0]
    return f"**Члан {broj}** (izvor: {p['izvor']}):\n\n{p['tekst']}", []


def handle_osoba_po_imenu(upit: str):
    """Pretraga osobe po imenu."""
    u = sredi_upit(upit)
    # Izvuci ime iz upita
    m = re.search(r'(?:ima li|ko je|pronadi|nadji)\s+([A-ZČĆŠĐŽ][a-zčćšđž]+(?:\s+[A-ZČĆŠĐŽ][a-zčćšđž]+)?)', upit)
    if not m:
        return "⚠️ Ne mogu da prepoznam ime.", []

    ime = m.group(1)
    docs = retrieve(upit, k=20)
    pogodci = []
    for d in docs:
        text = d.get("text", "")
        if ime.lower() in text.lower():
            pogodci.append(d)

    if not pogodci:
        return f"⚠️ {ime} — nisam pronašao u bazi.", []

    output = f"Pronađeno {len(pogodci)} rezultata:\n\n"
    for i, d in enumerate(pogodci[:5]):
        text = d.get("text", "")[:200]
        output += f"**[{i+1}]** {d.get('filename', '')}\n{text}...\n\n"
    return output, []


def handle_standard(upit: str, k: int = 10):
    """Standardni RAG."""
    docs = retrieve(upit, k=k)
    if not docs:
        return "⚠️ Nema rezultata.", 0, [], None

    kontekst = "\n\n---\n\n".join(d["text"] for d in docs[:5])

    # Probaj eksterni search
    ext_info = ""
    if detektuj_tip(upit) == "eksterno":
        ext = external_search(upit)
        if ext:
            ext_info = f"\n\n=== SPOLJNI IZVORI ===\n{ext}"

    # LLM
    try:
        llm = get_llm()
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"KONTEKST:\n{kontekst}{ext_info}\n\nPitanje: {upit}"},
        ]
        response = llm.invoke(messages)
        odgovor = response.content
    except Exception as e:
        odgovor = f"⚠️ LLM greška: {e}"

    return odgovor, len(docs), [], None


# ============================================================
# STREAMLIT UI
# ============================================================
st.set_page_config(page_title="Биро асистент v7", page_icon="🌲", layout="wide")

# Header
col_l, col_c, col_r = st.columns([1, 3, 1])
with col_c:
    st.image(LOGO_URL, width=110)
    st.markdown(
        "<h2 style='text-align: center; color: #1b4332; margin: 0;'>🌲 Биро асистент <small>v7 (LangChain)</small></h2>",
        unsafe_allow_html=True,
    )

st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("⚙️ Podešavanja")
    st.write(f"**Kolekcija:** `{TEXT_COLLECTION}`")

    if st.button("🔄 Osveži keš"):
        st.cache_resource.clear()
        st.rerun()

    if st.button("🗑️ Obriši razgovor"):
        st.session_state.messages = []
        st.rerun()

# 6 quick prompt dugmadi
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

# Pending pitanje
if st.session_state.pending:
    user_input = st.session_state.pending
    st.session_state.pending = None
else:
    user_input = st.chat_input("Postavi pitanje...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input, "images": []})

    with st.chat_message("user"):
        st.write(user_input)

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
                    odgovor, br_k, slike, _ = handle_standard(user_input)
                    if br_k and not meta:
                        meta = f"\n\n<sub>📊 Kandidati: {br_k}</sub>"
            except Exception as e:
                odgovor = f"⚠️ Greška: {e}"
                slike = []

            # Prikaz
            if slike:
                st.markdown("---")
                img_cols = st.columns(min(3, len(slike)))
                for idx, item in enumerate(slike):
                    with img_cols[idx % 3]:
                        if isinstance(item, tuple):
                            url, cap = item
                        else:
                            url, cap = item, ""
                        st.image(url, caption=cap, width=250)
                st.markdown("---")

            st.markdown(odgovor + meta)

    st.session_state.messages.append({
        "role": "assistant",
        "content": odgovor + meta,
        "images": slike,
    })

# Prikaz istorije
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg.get("images"):
            img_cols = st.columns(min(3, len(msg["images"])))
            for idx, item in enumerate(msg["images"]):
                with img_cols[idx % 3]:
                    if isinstance(item, tuple):
                        url, cap = item
                    else:
                        url, cap = item, ""
                    st.image(url, caption=cap, width=250)
