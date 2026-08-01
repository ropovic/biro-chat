"""
app.py v6.0 — Koristi strukturirana polja baze
================================================
KLJUČNO: Sada koristimo postojeća polja u payload-u:
  - fotografija_profil: "Funkcija" (direktor/zamenik/projektant)
  - dijagram: "ocr_tekst" (sadrži "ruža vetrova" itd.)
  - oprema: čist tekst tonera/štampača
  - pravni_akt: Kolektivni ugovor sa "члана N"

BEZ HALUCIJACIJA — čitamo TAČNO ono što piše u bazi.
"""

import os
import re
import streamlit as st
from qdrant_client import QdrantClient
from groq import Groq
from fastembed import TextEmbedding

# ============================================================
# CONFIG
# ============================================================
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
COLLECTION_NAME = os.environ.get("COLLECTION_NAME", "baza_cloud_v2_e5")
R2_PUBLIC_URL = "https://pub-49fb3cc788a74e0a9edbac7e11305b94.r2.dev"
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


def embed_query(text):
    if "e5" in EMBEDDING_MODEL.lower():
        text = f"query: {text}"
    return list(embed_model.embed([text]))[0].tolist()


def scroll_tip(tip, limit=200):
    """Skroluje zapise sa datim tipom. Bez filtera u Qdrant (nema indeks)."""
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
    return svi[:limit]


# ============================================================
# HANDLERI — direktno iz strukturiranih polja
# ============================================================
def handle_direktor():
    """Direktor: skeniraj fotografija_profil, nađi onaj gde Funkcija='direktor'."""
    points = scroll_tip("fotografija_profil", limit=100)
    direktori = []
    zamenici = []
    for p in points:
        payload = p.payload or {}
        tekst = payload.get("tekst", "") or ""
        url = payload.get("Link") or payload.get("slika_url", "")
        funkcija_raw = payload.get("Funkcija", "") or ""
        funkcija = funkcija_raw.lower()
        m = re.search(r'[Ff]otografij[ae]\s+([A-ZČĆŠĐŽ][a-zčćšđž]+\s+[A-ZČĆŠĐŽ][a-zčćšđž]+)', tekst)
        if not m:
            continue
        ime = m.group(1)
        # Detektuj precizno: "direktor" ali NE "pomoćnik" ili "zamenik"
        if "direktor" in funkcija and "zamenik" not in funkcija and "pomoćnik" not in funkcija:
            direktori.append((ime, url, funkcija_raw))
        elif "zamenik" in funkcija:
            zamenici.append((ime, url, funkcija_raw))

    if not direktori and not zamenici:
        return "⚠️ Nisu pronađeni direktor/zamenici u photo zapisima.", []

    slike = []
    delovi = []
    # Pokaži SAMO 1 direktor (prvi) + max 1 zamenik (prvi)
    if direktori:
        ime = direktori[0][0]
        url = direktori[0][1]
        delovi.append(f"**Direktor Biroа:** {ime}")
        if url:
            slike.append((url, f"Direktor: {ime}"))
    if zamenici:
        ime = zamenici[0][0]
        url = zamenici[0][1]
        delovi.append(f"**Zamenik direktora:** {ime}")
        if url:
            slike.append((url, f"Zamenik: {ime}"))
        # Ako ih ima više, samo imena
        if len(zamenici) > 1:
            ostali = ", ".join([z[0] for z in zamenici[1:]])
            delovi.append(f"Ostali zamenici: {ostali}")

    return "\n\n".join(delovi), slike


def handle_lista_zaposlenih():
    """Lista: izvuci sva imena iz 'Fotografija [Ime], [funkcija]'."""
    points = scroll_tip("fotografija_profil", limit=100)
    zaposleni = []
    for p in points:
        payload = p.payload or {}
        tekst = payload.get("tekst", "") or ""
        url = payload.get("Link") or payload.get("slika_url", "")
        funkcija = payload.get("Funkcija", "")
        m = re.search(r'[Ff]otografij[ae]\s+([A-ZČĆŠĐŽ][a-zčćšđž]+\s+[A-ZČĆŠĐŽ][a-zčćšđž]+)', tekst)
        if m:
            ime = m.group(1)
            zaposleni.append({"ime": ime, "url": url, "funkcija": funkcija})

    if not zaposleni:
        return "⚠️ Nisu pronađeni zaposleni u photo zapisima.", []

    # Dedupe po imenu
    seen = set()
    uniq = []
    for z in zaposleni:
        if z["ime"].lower() not in seen:
            seen.add(z["ime"].lower())
            uniq.append(z)

    imena_lista = "\n".join([f"• {z['ime']} — {z['funkcija']}" for z in uniq if z.get("funkcija")])
    if not imena_lista:
        imena_lista = "\n".join([f"• {z['ime']}" for z in uniq])

    slike = [(z["url"], z["ime"]) for z in uniq if z.get("url")]
    return f"**Запослени у Бироу ({len(uniq)}):**\n\n{imena_lista}", slike


def handle_oprema_specificno(upit):
    """Oprema: skeniraj 'oprema' tip, filtriraj po upitu."""
    points = scroll_tip("oprema", limit=50)
    u = upit.lower()
    is_toner = "toner" in u or "kertridž" in u or "kertridz" in u
    is_printer = "štampač" in u or "stampac" in u or "printer" in u

    delovi = []
    slike = []
    for p in points:
        payload = p.payload or {}
        tekst = (payload.get("tekst", "") or "").lower()
        izvor = payload.get("izvor", "") or payload.get("naziv_dokumenta", "")
        url = payload.get("slika_url", "")

        # Filtriranje
        if is_toner and not ("toner" in tekst or "kertrid" in tekst):
            continue
        if is_printer and not any(kw in tekst for kw in ["štampač", "stampac", "printer", "kyocera", "canon", "hp"]):
            continue

        # Skrati tekst
        cist = payload.get("tekst", "")
        # Izbaci adrese
        linije = []
        for l in cist.split("\n"):
            ll = l.lower()
            if any(x in ll for x in ["mihaila pupina", "birčaninova", "tel/fax",
                                      "javno preduzece", "trebovanje"]):
                continue
            if ll.strip():
                linije.append(l.strip())
        cist = " ".join(linije)[:300]

        if cist:
            delovi.append(f"**{izvor or 'Oprema'}:**\n{cist}")
        if url:
            slike.append((url, izvor or "Oprema"))

    if not delovi:
        return "⚠️ Nema pronađene opreme po tom upitu.", []
    return "**Pronađena oprema:**\n\n" + "\n\n---\n\n".join(delovi[:10]), slike


def handle_dijagram(upit):
    """Dijagram: skeniraj dijagram tip, koristi ocr_tekst za pretragu."""
    points = scroll_tip("dijagram", limit=200)
    u = upit.lower()

    delovi = []
    slike = []
    for p in points:
        payload = p.payload or {}
        tekst = (payload.get("tekst", "") or "").lower()
        ocr = (payload.get("ocr_tekst", "") or "").lower()
        izvor = payload.get("izvor", "") or ""
        url = payload.get("Link", "") or payload.get("slika_url", "")

        # Ako upit sadrži "ruža vetrova", filtriraj
        if "vetrova" in u or "ruza" in u:
            if not ("vetrova" in ocr or "vetrova" in tekst or "wind" in ocr or "ruza" in tekst):
                continue

        if url:
            slike.append((url, izvor or "Dijagram"))
        delovi.append(f"**{izvor or 'Dijagram'}**")

    if not slike:
        return "⚠️ Nema dijagrama koji odgovaraju tom upitu.", []
    return f"**Pronađeno {len(slike)} dijagrama:**", slike[:6]


def handle_clan(broj):
    """Pravni član: skeniraj pravni_akt, nađi 'član N'."""
    points = scroll_tip("pravni_akt", limit=500)
    pogodci = []
    for p in points:
        payload = p.payload or {}
        tekst = payload.get("tekst", "") or ""
        tekst_norm = tekst.lower()
        # Traži "члан N" ili "clan N"
        if f"члан {broj}" in tekst_norm or f"clan {broj}" in tekst_norm:
            # Izvuci oko člana
            idx = max(tekst_norm.find(f"члан {broј}"), tekst_norm.find(f"clan {broj}"))
            if idx < 0:
                continue
            start = max(0, idx - 50)
            end = min(len(tekst), idx + 800)
            izvor = payload.get("izvor", "") or payload.get("naziv_dokumenta", "")
            pogodci.append({"tekst": tekst[start:end], "izvor": izvor})

    if not pogodci:
        return f"⚠️ Član {broj} nije pronađen u bazi pravnih akata.", []

    # Dedupe
    seen = set()
    uniq = []
    for p in pogodci:
        kljuc = p["tekst"][:100].lower()
        if kljuc not in seen:
            seen.add(kljuc)
            uniq.append(p)
    p = uniq[0]
    return f"**Члан {broj}** (izvor: {p['izvor']}):\n\n{p['tekst']}", []


# ============================================================
# STANDARDNI RAG
# ============================================================
def do_rag(query, top_k=10):
    """Jednostavan RAG za opšta pitanja."""
    try:
        vec = embed_query(query)
        if hasattr(qdrant, "query_points"):
            response = qdrant.query_points(
                collection_name=COLLECTION_NAME,
                query=vec,
                limit=top_k,
            )
            points = response.points
        else:
            points = qdrant.search(
                collection_name=COLLECTION_NAME,
                query_vector=vec,
                limit=top_k,
            )
    except Exception as e:
        return "", 0, [], f"Greška: {e}"

    delovi = []
    slike = []
    seen = set()
    for hit in points:
        if not hit.payload:
            continue
        text = hit.payload.get("tekst", "") or hit.payload.get("text", "") or ""
        izvor = hit.payload.get("naziv_dokumenta", "") or hit.payload.get("izvor", "") or ""
        url = (hit.payload.get("Link", "") or hit.payload.get("slika_url", "") or
               hit.payload.get("image_url", "") or hit.payload.get("slika", ""))
        cist = re.sub(r'http[s]?://\S+', '', text).strip()
        cist = re.sub(r'\n{3,}', '\n\n', cist)
        cist = cist.replace("Ime Prezime", "[ime]")
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
    return kontekst, len(points), slike, ""


def ask_llm(messages):
    """Poziv LLM sa fallback."""
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
# RUTIRANJE
# ============================================================
def detektuj_tip(upit):
    u = upit.lower()
    # Direktor
    if "direktor" in u and "zamenik" not in u:
        return "direktor"
    if "ko je direktor" in u or "ko je novi direktor" in u:
        return "direktor"
    # Lista zaposlenih
    if any(kw in u for kw in ["svi zaposleni", "lista zaposlenih", "spisak", "ko radi", "kadrovska"]):
        return "lista_zaposlenih"
    # Štampači/toneri
    if "toner" in u or "kertrid" in u:
        return "oprema"
    if any(kw in u for kw in ["štampač", "stampac", "stampaci", "printer", "oprema"]):
        return "oprema"
    # Dijagram
    if any(kw in u for kw in ["dijagram", "mapa", "karta", "ruža vetrova", "ruza vetrova",
                                "vetrova", "grafikon", "šema", "shema", "tabela", "skica", "crtež"]):
        return "dijagram"
    # Pravni član
    m = re.search(r"član\s*(\d+)|clan\s*(\d+)", u)
    if m:
        broj = m.group(1) or m.group(2)
        return f"clan_{broj}"
    return "standard"


# ============================================================
# UI
# ============================================================
st.set_page_config(page_title="Биро асистент", page_icon="🌲", layout="wide")

# Header
col_l, col_c, col_r = st.columns([1, 3, 1])
with col_c:
    st.image(LOGO_URL, width=110)
    st.markdown(
        "<h2 style='text-align: center; color: #1b4332; margin: 0;'>🌲 Биро асистент</h2>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='text-align: center; color: #52796f; margin: 0;'>"
        "ПД „Србијашуме” • Биро за планирање</p>",
        unsafe_allow_html=True,
    )

st.markdown("---")

# 6 quick prompt dugmadi
st.markdown("##### 💡 Брза питања:")
cols = st.columns(6)
QUICK = [
    "Ко је директор Бироа?",
    "Који су запослени у Бироу?",
    "Који су штампачи у Бироу?",
    "Који тонeri се користе?",
    "Покажи дијаграм руже ветрова",
    "Члан 14",
]
for i, label in enumerate(QUICK):
    with cols[i]:
        if st.button(label, use_container_width=True, key=f"q{i}"):
            st.session_state.pending = label

st.markdown("---")

# Sidebar
with st.sidebar:
    st.image(LOGO_URL, width=80)
    st.markdown("### 🌲 Биро")
    st.caption(f"Kolekcija: {COLLECTION_NAME}")
    st.markdown("---")
    if st.button("🧹 Обриши разговор", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    if st.button("🔄 Освежи кеш", use_container_width=True):
        # Bezbedno: samo clear data cache, ne resource cache
        # Resource cache (qdrant, groq, embed) se ne dira — TTL to reguliše
        st.cache_data.clear()
        st.session_state.cache_msg = "Кеш података обрисан"
        st.rerun()
    if st.session_state.get("cache_msg"):
        st.success(f"✅ {st.session_state.cache_msg}")
        del st.session_state.cache_msg

# Istorija
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
        with st.spinner("Тражим у бази..."):
            try:
                tip = detektuj_tip(user_input)
                slike = []
                meta = ""

                if tip == "direktor":
                    odgovor, slike = handle_direktor()
                elif tip == "lista_zaposlenih":
                    odgovor, slike = handle_lista_zaposlenih()
                elif tip == "oprema":
                    odgovor, slike = handle_oprema_specificno(user_input)
                elif tip == "dijagram":
                    odgovor, slike = handle_dijagram(user_input)
                elif tip.startswith("clan_"):
                    broj = tip.split("_")[1]
                    odgovor, slike = handle_clan(broj)
                else:
                    kontekst, br_k, slike, err = do_rag(user_input)
                    if err:
                        st.error(err)
                        odgovor = err
                    else:
                        messages = [
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": f"KONTEKST:\n{kontekst}\n\nPitanje: {user_input}"},
                        ]
                        odgovor = ask_llm(messages)
                        meta = f"\n\n<sub>📊 Kandidati: {br_k}</sub>"

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
