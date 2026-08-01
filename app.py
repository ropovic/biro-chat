"""
app.py v5.0 — MINIMALNA verzija
================================
Bez specijalnih handlera. Samo:
- Qdrant search (bez filtera)
- Embed + LLM
- Slike gde postoje
- 6 quick prompt dugmadi

Ovo je vraćanje na osnove — verzija koja je radila pre svih dorada.
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
    "Ne izvodi zaključke, ne pretpostavljaj. "
    "Navedi imena zaposlenih SAMO ako su eksplicitno u kontekstu. "
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


# ============================================================
# RAG — jednostavan
# ============================================================
def do_rag(query, top_k=10):
    """Jednostavan RAG: Qdrant search + kontekst + slike."""
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
        return "", 0, [], f"Greška pri pretrazi: {e}"

    delovi = []
    slike = []
    seen = set()

    for hit in points:
        if not hit.payload:
            continue
        text = hit.payload.get("tekst", "") or hit.payload.get("text", "") or ""
        izvor = hit.payload.get("naziv_dokumenta", "") or hit.payload.get("file_name", "") or ""
        url = (hit.payload.get("Link", "") or hit.payload.get("slika_url", "") or
               hit.payload.get("image_url", "") or hit.payload.get("slika", ""))

        # Skrati tekst
        cist = re.sub(r'http[s]?://\S+', '', text).strip()
        cist = re.sub(r'\n{3,}', '\n\n', cist)
        if "Ime Prezime" in cist:
            cist = cist.replace("Ime Prezime", "[ime]")
        if len(cist) > 600:
            cist = cist[:600] + "..."
        if cist:
            delovi.append(f"[{izvor}]\n{cist}")

        # Slike
        if url and url.startswith("http") and url not in seen:
            slike.append((url, izvor or "Slika"))
            seen.add(url)

    kontekst = "\n\n---\n\n".join(delovi)
    if len(kontekst) > 6000:
        kontekst = kontekst[:6000] + "\n[Skraćeno]"

    return kontekst, len(points), slike, ""


def ask_llm(messages):
    """Poziv LLM sa fallback-om na manji model."""
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
    "Члан 14 колективног уговора",
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
        st.cache_resource.clear()
        st.cache_data.clear()
        st.session_state.cache_msg = "Кеш обрисан"
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
                kontekst, br_k, slike, err = do_rag(user_input)
                if err:
                    st.error(err)
                    st.session_state.messages.append({
                        "role": "assistant", "content": err, "images": []
                    })
                else:
                    messages = [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": f"KONTEKST:\n{kontekst}\n\nPitanje: {user_input}"},
                    ]
                    odgovor = ask_llm(messages)

                    # Slike
                    if slike:
                        st.markdown("---")
                        img_cols = st.columns(min(3, len(slike)))
                        for idx, (url, cap) in enumerate(slike):
                            with img_cols[idx % 3]:
                                st.image(url, caption=cap, width=250)
                        st.markdown("---")

                    st.markdown(odgovor)
                    st.caption(f"📊 Kandidati: {br_k}")

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": odgovor,
                        "images": slike,
                    })
            except Exception as e:
                st.error(f"⚠️ Greška: {e}")
                st.session_state.messages.append({
                    "role": "assistant", "content": f"⚠️ Greška: {e}", "images": []
                })
