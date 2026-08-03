"""
app.py v13 — bez sentence_transformers, koristi fastembed i MPNet za sve
"""

import os
import re
import time
import streamlit as st
from qdrant_client import QdrantClient, models
from groq import Groq
from fastembed import TextEmbedding
import requests

# ============================================================
# CONFIG
# ============================================================
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
COLLECTION_NAME = os.environ.get("COLLECTION_NAME", "baza_cloud_v4")
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

def scroll_tip(tip, limit=200):
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

def scroll_all(limit=500):
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
    return svi[:limit]

def search_text(query, top_k=10, tip_filter=None):
    vec = embed_query(query)
    filter_cond = None
    if tip_filter:
        filter_cond = models.Filter(
            must=[models.FieldCondition(key="tip", match=models.MatchValue(value=tip_filter))]
        )
    try:
        return qdrant_search(
            collection_name=COLLECTION_NAME,
            query_vector=vec,
            limit=top_k,
            query_filter=filter_cond,
            with_payload=True,
        )
    except Exception as e:
        st.error(f"Greška pri pretrazi: {e}")
        return []

# ============================================================
# HANDLERI
# ============================================================

def handle_direktor():
    points = scroll_tip("fotografija_profil", limit=100)
    if not points:
        hits = search_text("direktor Biroa", top_k=5)
        for hit in hits:
            payload = hit.payload or {}
            tekst = payload.get("tekst", "")
            if "direktor" in tekst.lower():
                m = re.search(r'direktor\s+([A-ZČĆŠĐŽ][a-zčćšđž]+\s+[A-ZČĆŠĐŽ][a-zčćšđž]+)', tekst, re.IGNORECASE)
                if m:
                    ime = m.group(1)
                    return f"**Direktor Biroa:** {ime}", []
        return "⚠️ Nisu pronađeni direktor/zamenici u bazi.", []

    direktori = []
    zamenici = []
    for p in points:
        payload = p.payload or {}
        tekst = payload.get("tekst", "") or ""
        url = payload.get("slika_url", "") or ""
        funkcija_raw = payload.get("Funkcija", "") or payload.get("employee_name", "")
        if not funkcija_raw:
            m = re.search(r'(?:Funkcija|funkcija)\s*[:;]\s*([^,.\n]+)', tekst, re.IGNORECASE)
            if m:
                funkcija_raw = m.group(1).strip()
        funkcija = funkcija_raw.lower()
        m = re.search(r'[Ff]otografij[ae]\s+([A-ZČĆŠĐŽ][a-zčćšđž]+\s+[A-ZČĆŠĐŽ][a-zčćšđž]+)', tekst)
        if not m:
            ime = payload.get("employee_name", "")
            if not ime:
                continue
        else:
            ime = m.group(1)

        if "direktor" in funkcija and "zamenik" not in funkcija and "pomoćnik" not in funkcija:
            direktori.append((ime, url, funkcija_raw))
        elif "zamenik" in funkcija:
            zamenici.append((ime, url, funkcija_raw))

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
    points = scroll_tip("fotografija_profil", limit=100)
    if not points:
        return "⚠️ Nisu pronađeni zaposleni.", []
    zaposleni = []
    for p in points:
        payload = p.payload or {}
        tekst = payload.get("tekst", "") or ""
        url = payload.get("slika_url", "") or ""
        ime = payload.get("employee_name", "")
        if not ime:
            m = re.search(r'[Ff]otografij[ae]\s+([A-ZČĆŠĐŽ][a-zčćšđž]+\s+[A-ZČĆŠĐŽ][a-zčćšđž]+)', tekst)
            if m:
                ime = m.group(1)
        if ime:
            funkcija = payload.get("Funkcija", "")
            if not funkcija:
                fm = re.search(r'(?:Funkcija|funkcija)\s*[:;]\s*([^,.\n]+)', tekst, re.IGNORECASE)
                if fm:
                    funkcija = fm.group(1).strip()
            zaposleni.append({"ime": ime, "url": url, "funkcija": funkcija})
    if not zaposleni:
        return "⚠️ Nisu pronađeni zaposleni.", []
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
    return f"**Zaposleni u Birou ({len(uniq)}):**\n\n{imena_lista}", slike

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
    points = scroll_tip("fotografija_profil", limit=200)
    if not points:
        return f"⚠️ Nema fotografija u bazi.", []
    pogodci = []
    search_term_lower = search_term.lower()
    for p in points:
        payload = p.payload or {}
        tekst = payload.get("tekst", "") or ""
        employee_name = payload.get("employee_name", "") or ""
        url = payload.get("slika_url", "") or ""
        if employee_name:
            if search_term_lower in employee_name.lower():
                ime = employee_name
                funkcija = payload.get("Funkcija", "")
                if not funkcija:
                    fm = re.search(r'(?:Funkcija|funkcija)\s*[:;]\s*([^,.\n]+)', tekst, re.IGNORECASE)
                    if fm:
                        funkcija = fm.group(1).strip()
                if not any(p["ime"].lower() == ime.lower() for p in pogodci):
                    pogodci.append({"ime": ime, "url": url, "funkcija": funkcija})
                continue
        if search_term_lower in tekst.lower():
            m = re.search(r'[Ff]otografij[ae]\s+([A-ZČĆŠĐŽ][a-zčćšđž]+\s+[A-ZČĆŠĐŽ][a-zčćšđž]+)', tekst)
            if m:
                ime = m.group(1)
            else:
                continue
            funkcija = payload.get("Funkcija", "")
            if not funkcija:
                fm = re.search(r'(?:Funkcija|funkcija)\s*[:;]\s*([^,.\n]+)', tekst, re.IGNORECASE)
                if fm:
                    funkcija = fm.group(1).strip()
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
    points = scroll_tip("oprema", limit=100)
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
            return "⚠️ Nisu pronađeni toneri.", []
        delovi = ["**Toneri u Birou:**", ""]
        for t in sorted(toneri_lista):
            delovi.append(f"- {dodaj_proizvodjaca(t)}")
        return "\n".join(delovi), []
    if is_printer:
        if not stampaci_lista:
            return "⚠️ Nisu pronađeni štampači.", []
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
    hits = search_text(upit, top_k=10, tip_filter="dijagram")
    if not hits:
        hits = search_text(upit, top_k=10, tip_filter="image")
    if not hits:
        points = scroll_tip("dijagram", limit=10)
        if not points:
            points = scroll_tip("image", limit=10)
        if points:
            slike = []
            for p in points:
                url = p.payload.get("slika_url", "") or ""
                if url and url.startswith("http"):
                    naziv = p.payload.get("naziv_dokumenta", "") or "Dijagram"
                    slike.append((url, naziv))
            if slike:
                return f"**Nema slika za '{upit}', ali evo drugih dijagrama:**", slike[:6]
        return "⚠️ Nema pronađenih dijagrama.", []
    slike = []
    for hit in hits:
        payload = hit.payload
        url = payload.get("slika_url", "") or ""
        if url and url.startswith("http"):
            naziv = payload.get("naziv_dokumenta", "") or "Dijagram"
            slike.append((url, naziv))
    if not slike:
        return "⚠️ Pronađene slike nemaju javni URL.", []
    return f"**Pronađeno {len(slike)} slika/dijagrama:**", slike[:6]

def handle_clan(broj):
    all_points = scroll_all(limit=500)
    pogodci = []
    clan_pat = re.compile(
        rf'(?:clan|clana|clanom|clanu|clane|cl\.|cln\.)\s*{re.escape(broj)}\b(.*?)'
        rf'(?=(?:clan|clana|clanom|clanu|clane|cl\.|cln\.)\s*\d+\b|$)',
        re.DOTALL | re.IGNORECASE
    )
    clan_check = re.compile(
        rf'(?:clan|clana|clanom|clanu|clane|cl\.|cln\.)\s*{re.escape(broj)}\b',
        re.IGNORECASE
    )
    for p in all_points:
        payload = p.payload or {}
        tekst = payload.get("tekst", "") or ""
        tekst_norm = sredi_upit(tekst)
        if not clan_check.search(tekst_norm):
            continue
        for m in clan_pat.finditer(tekst):
            clan_tekst = m.group(0).strip()
            if len(clan_tekst) < 30:
                continue
            if len(clan_tekst) > 8000:
                clan_tekst = clan_tekst[:8000] + "\n...[Skraćeno]"
            izvor = payload.get("naziv_dokumenta", "") or ""
            pogodci.append({"tekst": clan_tekst, "izvor": izvor, "duzina": len(clan_tekst)})
        if not pogodci and clan_check.search(tekst_norm):
            for cm in clan_check.finditer(tekst_norm):
                start_orig = max(0, cm.start() - 30)
                end_orig = min(len(tekst), cm.end() + 3500)
                clan_tekst = tekst[start_orig:end_orig].strip()
                if len(clan_tekst) >= 30:
                    pogodci.append({
                        "tekst": clan_tekst,
                        "izvor": payload.get("naziv_dokumenta", "") or "",
                        "duzina": len(clan_tekst),
                    })
                    break
    if not pogodci:
        return f"⚠️ Član {broj} nije pronađen.", []
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
        izvor = hit.payload.get("naziv_dokumenta", "") or ""
        url = hit.payload.get("slika_url", "") or ""
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
    ekstern = ["ko je ministar", "ko je predsednik", "sta je", "sta znaci", "koji je", "kako se zove",
               "gde se nalazi", "kada je", "koliko kosta", "pd srbijasume", "javno preduzece",
               "o kompaniji", "istorija", "sediste", "kontakt", "veb sajt"]
    return any(kw in u for kw in ekstern)

def je_pitanje_o_kompaniji(upit):
    u = sredi_upit(upit)
    kompanija = ["pd srbijasume", "javno preduzece", "preduzece", "firma", "kompanija", "organizacija", "istorija"]
    return any(kw in u for kw in kompanija)

def detektuj_tip(upit):
    u = sredi_upit(upit)
    if any(kw in u for kw in ["opiši sliku", "šta je na slici", "šta prikazuje"]):
        return "vizuelna_analiza"
    m = re.search(r'(?:clan|clana|clanom|clanu|clane|cl\.|cln\.)\s*(\d+)', u)
    if m:
        return f"clan_{m.group(1)}"
    if "direktor" in u and "zamenik" not in u:
        return "direktor"
    words = u.split()
    if len(words) <= 3 and not any(kw in u for kw in ["stampac", "toner", "oprema", "dijagram", "mapa", "karta"]):
        return "osoba_ime"
    if "zaposlen" in u:
        if any(kw in u for kw in ["svi", "lista", "spisak", "koji su", "ko je sve", "navedi"]):
            return "lista_zaposlenih"
    if "toner" in u or "kertrid" in u:
        return "oprema"
    if any(kw in u for kw in ["stampac", "stampaci", "printer", "pisač", "oprema", "racunar"]):
        return "oprema"
    if any(kw in u for kw in ["dijagram", "mapa", "karta", "ruza vetrova", "grafikon", "sema", "tabela", "skica"]):
        return "dijagram"
    return "standard"

# ============================================================
# STREAMLIT UI (skraćen)
# ============================================================
st.set_page_config(page_title="🌲 Biro asistent", page_icon="🌲", layout="wide")
col_l, col_c, col_r = st.columns([1, 3, 1])
with col_c:
    st.image(LOGO_URL, width=110)
    st.markdown("<h2 style='text-align: center; color: #1b4332;'>🌲 Biro asistent</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #52796f;'>PD „Srbijašume” • Biro za planiranje</p>", unsafe_allow_html=True)
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
                    hits = search_text(user_input, top_k=1, tip_filter="dijagram")
                    if not hits:
                        hits = search_text(user_input, top_k=1, tip_filter="fotografija_profil")
                    if hits:
                        url = hits[0].payload.get("slika_url", "") or ""
                        if url:
                            try:
                                import base64, io
                                from PIL import Image
                                response = requests.get(url, timeout=10)
                                img = Image.open(io.BytesIO(response.content))
                                # Ovde bi išao poziv HF API-ju, ali preskačemo radi jednostavnosti
                                odgovor = f"Analiza slike nije implementirana zbog konflikta zavisnosti."
                            except:
                                odgovor = "⚠️ Vizuelna analiza nije dostupna."
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
                        messages = [
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": f"KONTEKST IZ BAZE:\n{kontekst}\n\nPitanje: {user_input}"},
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
                st.session_state.messages.append({"role": "assistant", "content": odgovor + meta, "images": slike})
            except Exception as e:
                st.error(f"⚠️ Greška: {e}")
                st.session_state.messages.append({"role": "assistant", "content": f"⚠️ Greška: {e}", "images": []})