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


def handle_osoba_po_imenu(upit):
    """Traži osobu po imenu u photo records."""
    import re as _re
    # Izvuci imena (veliko slovo + veliko slovo) iz ORIGINALNOG upita
    imena = _re.findall(r'\b[A-ZČĆŠĐŽ][a-zčćšđž]{2,}\s+[A-ZČĆŠĐŽ][a-zčćšđž]{2,}\b', upit or "")
    # ćirilica varijanta
    if not imena:
        imena = _re.findall(r'\b[А-ЯЁ][а-яё]{2,}\s+[А-ЯЁ][а-яё]{2,}\b', upit or "")
    if not imena:
        return "⚠️ Nisam pronašao ime u pitanju. Koristite format: 'Ime Prezime'.", []

    points = scroll_tip("fotografija_profil", limit=100)
    pogodci = []
    for p in points:
        payload = p.payload or {}
        tekst = payload.get("tekst", "") or ""
        tekst_norm = sredi_upit(tekst)
        url = payload.get("Link") or payload.get("slika_url", "")
        for ime in imena:
            ime_norm = sredi_upit(ime)
            # Podeli ime na delove i traži sve delove
            delovi_imena = ime_norm.split()
            if all(d in tekst_norm for d in delovi_imena):
                # Izvuci lepo ime
                m = _re.search(r'[Ff]otografij[ae]\s+([A-ZČĆŠĐŽ][a-zčćšđž]+\s+[A-ZČĆŠĐŽ][a-zčćšđž]+)', tekst)
                ime_lepo = m.group(1) if m else ime
                funkcija = payload.get("Funkcija", "")
                pogodci.append({"ime": ime_lepo, "url": url, "funkcija": funkcija})

    if not pogodci:
        return f"⚠️ {', '.join(imena)} — nisam pronašao u bazi.", []

    # Dedupe
    seen = set()
    uniq = []
    for p in pogodci:
        if p["ime"] not in seen:
            seen.add(p["ime"])
            uniq.append(p)

    delovi = [f"**Pronađeno {len(uniq)}:**", ""]
    for p in uniq:
        if p.get("funkcija"):
            delovi.append(f"- **{p['ime']}** — {p['funkcija']}")
        else:
            delovi.append(f"- **{p['ime']}**")
    slike = [(p["url"], p["ime"]) for p in uniq if p["url"]]
    return "\n".join(delovi), slike


def je_pitanje_za_eksterno(upit):
    """Da li pitanje zahteva eksternu pretragu (opšta znanja)."""
    u = sredi_upit(upit)
    # Pitanja o ljudima, definicijama, opštim pojmovima, KOMPANIJI
    ekstern_ključne = [
        # Pitanja o ljudima
        "ko je ministar", "ko je predsednik", "ko je direktor pd",
        "ko je osnivac", "ko je osnovao", "ko je izumeo", "ko je napravio",
        # Definicije
        "sta je", "sta su", "sta znaci", "sta predstavlja",
        # Opšta pitanja
        "koji je", "koja je", "koje je",
        "kako se zove", "gde se nalazi", "kada je",
        "koliko kosta", "koliko je",
        # KOMPANIJA / FIRMA
        "pd srbijasume", "javno preduzece", "preduzece za gazdovanje",
        "o kompaniji", "o firmi", "istorija", "kako posluje",
        "sediste", "kontakt", "veb sajt", "web sajt", "sajt",
    ]
    return any(kw in u for kw in ekstern_ključne)


def je_pitanje_o_kompaniji(upit):
    """Da li pitanje je o kompaniji (PD Srbijašume) — ne treba slike zaposlenih."""
    u = sredi_upit(upit)
    kompanija_ključne = [
        "pd srbijasume", "javno preduzece", "preduzece", "firma",
        "kompanija", "organizacija", "istorija", "delatnost",
        "sediste", "veb sajt", "web sajt",
    ]
    return any(kw in u for kw in kompanija_ključne)


def handle_oprema_specificno(upit):
    """Oprema: skeniraj 'oprema' tip, prikaži čistu listu.
    Filter: štampači ILI toneri (ne oba).
    Razdvaja prema tipu zapisa, ne samo keyword match-u."""
    points = scroll_tip("oprema", limit=50)
    u = sredi_upit(upit)
    is_toner = "toner" in u or "kertridz" in u
    is_printer = any(kw in u for kw in ["stampac", "stampaci", "printer", "pisac"])

    # Specifični patterni za štampače (isključuju tonere)
    # Kyocera FS-9530dn, M3655idn, P2040dn, TASKalfa, ECOSYS
    # HP Designjet, OfficeJet, LaserJet, PageWide
    # Canon imageRUNNER, TX-3000, iR, PIXMA
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

    # Specifični patterni za tonere (isključuju štampače)
    # Kyocera TK-XXX, HP CXXXX ili CE-XXX, Canon PFI-XXX ili CL-XXX
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
        tekst = sredi_upit(tekst_orig)

        # Detektuj šta ovaj zapis sadrži
        has_printer = bool(printer_pat.search(tekst_orig))
        has_toner = bool(toner_pat.search(tekst_orig))

        if is_printer:
            # Uzimamo SAMO ako zapis sadrži printer (ne toner)
            if not has_printer:
                continue
            for m in printer_pat.findall(tekst_orig):
                stampaci_lista.add(m.strip())
        elif is_toner:
            # Uzimamo SAMO ako zapis sadrži toner (ne printer)
            if not has_toner:
                continue
            for m in toner_pat.findall(tekst_orig):
                toneri_lista.add(m.strip())

    def dodaj_proizvodjaca(kod):
        if kod.upper().startswith("TK-"):
            return f"Kyocera {kod}"
        if kod.upper().startswith("CE"):
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

    # Opšti upit — oba
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
    """Dijagram: skeniraj dijagram tip, koristi ocr_tekst za pretragu."""
    points = scroll_tip("dijagram", limit=200)
    u = sredi_upit(upit)

    # Ako korisnik traži specifično (ruža vetrova), pokušaj prvo sa filterom
    if any(kw in u for kw in ["vetrova", "ruza vetrova", "wind"]):
        filtrirane = []
        for p in points:
            payload = p.payload or {}
            tekst = sredi_upit(payload.get("tekst", "") or "")
            ocr = sredi_upit(payload.get("ocr_tekst", "") or "")
            url = payload.get("Link", "") or payload.get("slika_url", "")
            if ("vetrova" in ocr or "vetrova" in tekst or "wind" in ocr
                or "ruza" in tekst or "pravac" in tekst):
                if url and url.startswith("http"):
                    filtrirane.append((url, payload.get("izvor", "") or "Dijagram"))
        if filtrirane:
            return f"**Pronađeno {len(filtrirane)} dijagrama vetrova:**", filtrirane[:6]

    # Ako nema specifičnog filtera, vrati sve dostupne dijagrame
    slike = []
    for p in points:
        payload = p.payload or {}
        url = payload.get("Link", "") or payload.get("slika_url", "")
        if url and url.startswith("http"):
            slike.append((url, payload.get("izvor", "") or "Dijagram"))

    if not slike:
        return "⚠️ Nema dijagrama u bazi.", []
    return f"**Pronađeno {len(slike)} dijagrama:**", slike[:6]


def handle_clan(broj):
    """Pravni član: skeniraj pravni_akt, nađi CEO clan N do sledećeg clana ili kraja.
    Podržava: clan, clana, clanom, clanu, clane, cl., cln. (svi padeži)"""
    points = scroll_tip("pravni_akt", limit=500)
    pogodci = []
    # Regex pokriva SVE padeže + skraćenice
    clan_pat = re.compile(
        rf'(?:clan|clana|clanom|clanu|clane|cl\.|cln\.)\s*{re.escape(broj)}\b(.*?)'
        rf'(?=(?:clan|clana|clanom|clanu|clane|cl\.|cln\.)\s*\d+\b|$)',
        re.DOTALL | re.IGNORECASE
    )
    # Provera da li uopšte postoji bilo koja varijanta člana
    clan_check = re.compile(
        rf'(?:clan|clana|clanom|clanu|clane|cl\.|cln\.)\s*{re.escape(broj)}\b',
        re.IGNORECASE
    )

    for p in points:
        payload = p.payload or {}
        tekst = payload.get("tekst", "") or ""
        tekst_norm = sredi_upit(tekst)
        if not clan_check.search(tekst_norm):
            continue
        # Nađi SVE matcheve (član se može ponoviti u dokumentu)
        for m in clan_pat.finditer(tekst):
            clan_tekst = m.group(0).strip()
            if len(clan_tekst) < 30:
                continue
            if len(clan_tekst) > 4000:
                clan_tekst = clan_tekst[:4000] + "\n...[Skraćeno]"
            izvor = payload.get("izvor", "") or payload.get("naziv_dokumenta", "")
            pogodci.append({"tekst": clan_tekst, "izvor": izvor, "duzina": len(clan_tekst)})

    if not pogodci:
        return f"⚠️ Član {broj} nije pronađen u bazi pravnih akata.", []

    # Dedupe + uzmi najduži (kompletniji)
    seen = set()
    uniq = []
    for p in pogodci:
        kljuc = sredi_upit(p["tekst"][:200])
        if kljuc not in seen:
            seen.add(kljuc)
            uniq.append(p)
    # Sortiraj po dužini (najduži = najkompletniji)
    uniq.sort(key=lambda x: -x["duzina"])
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
# EKSTERNI SEARCH (Tavily) — fallback kad baza nema podatak
# ============================================================
import requests

def external_search(query, max_results=3):
    """Tavily pretraga. Vraća formatiran tekst ili None ako ne radi.
    Besplatno: 1000 pretraga/mesec. signup: https://tavily.com"""
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


def je_kontekst_dovoljan(kontekst):
    """Provera da li kontekst ima dovoljno smislenog sadržaja."""
    if not kontekst or len(kontekst.strip()) < 100:
        return False
    # Ako sadrži samo "nije pronađeno" ili je premali
    niske_reci = ["nije pronađeno", "nema podatak", "podatak nije"]
    tekst_lower = kontekst.lower()
    if any(fraza in tekst_lower for fraza in niske_reci) and len(kontekst) < 300:
        return False
    return True


# ============================================================
# RUTIRANJE
# ============================================================
def sredi_upit(t):
    """Konvertuje ćirilicu u latinicu i uklanja dijakritike."""
    if not t:
        return ""
    # 1) Ćirilica → latinica
    zamene_cir = {
        'а':'a','б':'b','в':'v','г':'g','д':'d','е':'e','ж':'ž','з':'z','и':'i',
        'ј':'j','к':'k','л':'l','м':'m','н':'n','о':'o','п':'p','р':'r','с':'s',
        'т':'t','у':'u','ф':'f','х':'h','ц':'c','ч':'č','ш':'š','ć':'ć','ђ':'đ',
    }
    out1 = []
    for c in str(t).lower():
        out1.append(zamene_cir.get(c, c))
    s = "".join(out1)
    # 2) Ukloni dijakritike
    zamene_dij = {'č':'c', 'ć':'c', 'š':'s', 'ž':'z', 'đ':'d'}
    return "".join([zamene_dij.get(c, c) for c in s])


def detektuj_tip(upit):
    u = sredi_upit(upit)  # KLJUČNO: ćirilica → latinica

    # Direktor (pre liste, jer "direktor" sadrži specifičniji pojam)
    if "direktor" in u and "zamenik" not in u:
        return "direktor"

    # Traženje osobe po imenu: "pronađi [Ime]", "nađi [Ime]", "ima li [Ime]"
    if any(kw in u for kw in ["pronadi", "nadji", "nadjem",
                                "ima li", "gde je", "ko je to", "sta je sa"]):
        # Ali NE ako je o direktoru ili listi zaposlenih
        if "direktor" not in u and "zaposleni" not in u:
            return "osoba_ime"

    # Lista zaposlenih — "zaposleni" + indikatori liste/pitanja
    if "zaposlen" in u:
        indikatori = ["svi", "lista", "spisak", "koji su", "ko je sve", "navedi",
                      "ko radi", "kadrov", "imena", "ljudi", "ko je", "tko je"]
        if any(ind in u for ind in indikatori):
            return "lista_zaposlenih"

    # Toneri (specifičniji od štampača)
    if "toner" in u or "kertrid" in u:
        return "oprema"

    # Štampači
    if any(kw in u for kw in ["stampac", "stampaci", "printer", "pisač", "oprema", "racunar"]):
        return "oprema"

    # Dijagram / vizuel
    if any(kw in u for kw in ["dijagram", "mapa", "karta", "ruza vetrova", "vetrova",
                                "grafikon", "sema", "shema", "tabela", "skica",
                                "crtez", "prikaz", "pokaz"]):
        return "dijagram"

    # Pravni član
    m = re.search(r"clan\s*(\d+)", u)
    if m:
        return f"clan_{m.group(1)}"

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
                elif tip == "osoba_ime":
                    odgovor, slike = handle_osoba_po_imenu(user_input)
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
                        # Za opšta pitanja (Ko je ministar, Šta je X) — probaj eksterno
                        ext_info = ""
                        koristio_ext = False
                        if je_pitanje_za_eksterno(user_input):
                            ext = external_search(user_input)
                            if ext:
                                ext_info = f"\n\n=== SPOLJNI IZVORI ===\n{ext}"
                                koristio_ext = True
                        # Za pitanja o KOMPANIJI — ukloni slike zaposlenih iz baze
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
