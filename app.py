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
    """Traži osobu po imenu u photo records.
    Podržava: 'Ime Prezime', 'Ime', 'Ime Prezime' (genitiv)."""
    import re as _re
    if not upit:
        return "⚠️ Prazno pitanje.", []

    # 1) Probaj dvo-člano ime (npr. "Bojana Jelić")
    imena = _re.findall(r'\b[A-ZČĆŠĐŽ][a-zčćšđž]{2,}\s+[A-ZČĆŠĐŽ][a-zčćšđž]{2,}\b', upit)
    # 2) Probaj ćirilično dvo-člano ime
    if not imena:
        imena = _re.findall(r'\b[А-ЯЁ][а-яё]{2,}\s+[А-ЯЁ][а-яё]{2,}\b', upit)
    # 3) Probaj jedno-člano ime (npr. "Bojane", "Bojana", "petrovića")
    if not imena:
        # Filtriraj "stop reči" da ne bismo uzeli "biro", "srbijasume" itd.
        stop_reci_ime = {"biro", "biroa", "srbijasume", "srbija", "suma", "sumama",
                         "baze", "birou", "kolektivni", "ugovor", "clan", "preduzece",
                         "firma", "kompanija", "pd", "jp",
                         "svi", "sve", "sva", "kako", "sta", "koji", "koja", "koje",
                         "gde", "kada", "imam", "imaju", "postoji", "treba", "hocu",
                         "ovaj", "taj", "ovo", "ta", "to", "neka", "neko", "nesto",
                         "moze", "molim", "zasto", "zbog", "prema", "preko"}
        # CASE-INSENSITIVE: bilo veliko ili malo slovo na početku
        kandidati = _re.findall(r'\b[A-Za-zčćšđžČĆŠĐŽ][a-zčćšđž]{3,}\b', upit or "")
        # Filtriraj stop reči (case-insensitive)
        kandidati = [k for k in kandidati if sredi_upit(k) not in stop_reci_ime]
        if kandidati:
            imena = kandidati  # npr. ["petrovića"] — case-insensitive matching će raditi

    if not imena:
        return ("⚠️ Nisam pronašao ime u pitanju.\n\n"
                "Koristite format: **'Ime Prezime'** (npr. 'Bojana Jelić') "
                "ili **'Ime'** (npr. 'Bojana')."), []

    # Normalizuj imena za pretragu
    imena_za_pretragu = []
    for ime in imena:
        ime_norm = sredi_upit(ime)
        # Podeli na delove
        delovi = ime_norm.split()
        imena_za_pretragu.append({
            "originalno": ime,
            "norm": ime_norm,
            "delovi": delovi,
        })

    points = scroll_tip("fotografija_profil", limit=100)
    pogodci = []
    for p in points:
        payload = p.payload or {}
        tekst = payload.get("tekst", "") or ""
        tekst_norm = sredi_upit(tekst)
        url = payload.get("Link") or payload.get("slika_url", "")

        for ime_info in imena_za_pretragu:
            # WORD BOUNDARY matching: koristimo regex \b za svaki deo
            svi_delovi_prisutni = True
            for d in ime_info["delovi"]:
                if not _re.search(rf'\b{_re.escape(d)}', tekst_norm):
                    svi_delovi_prisutni = False
                    break
            if not svi_delovi_prisutni:
                continue

            # Izvuci lepo ime iz "Fotografija [Ime Prezime], ..."
            m = _re.search(
                r'[Ff]otografij[ae]\s+([A-ZČĆŠĐŽ][a-zčćšđž]+(?:\s+[A-ZČĆŠĐŽ][a-zčćšđž]+)?)',
                tekst
            )
            ime_lepo = m.group(1) if m else ime_info["originalno"]
            funkcija = payload.get("Funkcija", "")
            # Dedupe po IMENU_LEPOM (lowercase)
            kljuc = sredi_upit(ime_lepo)
            pogodci.append({
                "ime": ime_lepo,
                "url": url,
                "funkcija": funkcija,
                "kljuc": kljuc,
            })

    if not pogodci:
        imena_str = ", ".join([i["originalno"] for i in imena_za_pretragu])
        return f"⚠️ **{imena_str}** — nisam pronašao u bazi fotografija.", []

    # Dedupe po kljuc-u (lowercase imena)
    seen = set()
    uniq = []
    for p in pogodci:
        if p["kljuc"] not in seen:
            seen.add(p["kljuc"])
            uniq.append(p)

    # KLJUČNO: Ako je samo jedno ime traženo, vrati SAMO njegove slike
    # (ne i slike drugih koji su možda slučajno matchovali)
    # TOLERIŠE PADEŽE: "Bojana" matchuje "Bojane" (prvih 4+ slova)
    if len(imena_za_pretragu) == 1:
        target_parts = imena_za_pretragu[0]["delovi"]
        filtered = []
        for p in uniq:
            # Za svaki deo traženog imena, proveri da li postoji reč u ime_lepo
            # koja počinje sa prva 4+ slova
            words = p["kljuc"].split()
            all_match = True
            for d in target_parts:
                if len(d) < 4:
                    # Kratka reč — tačno podudaranje
                    if not any(w == d for w in words):
                        all_match = False
                        break
                else:
                    # Duga reč — podudaranje po prefiksu (4+ slova)
                    prefiks = d[:max(4, len(d) - 1)]
                    if not any(w.startswith(prefiks) for w in words):
                        all_match = False
                        break
            if all_match:
                filtered.append(p)
        uniq = filtered if filtered else uniq

    delovi_text = [f"**Pronađeno {len(uniq)}:**", ""]
    for p in uniq:
        if p.get("funkcija"):
            delovi_text.append(f"- **{p['ime']}** — {p['funkcija']}")
        else:
            delovi_text.append(f"- **{p['ime']}**")
    # Maksimalno 3 slike
    slike = [(p["url"], p["ime"]) for p in uniq if p["url"]][:3]
    return "\n".join(delovi_text), slike


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
    """Dijagram: skroluje dijagram tip, koristi vizuel_opis/ocr_tekst za pretragu.
    Filtrira logo zapise. Koristi vizuel_opis ako postoji (Vision/heuristika)."""
    points = scroll_tip("dijagram", limit=200)
    u = sredi_upit(upit)

    # Helper: da li je zapis LOGO
    def je_logo(payload):
        tekst = (payload.get("tekst", "") or payload.get("Objekat", "") or "").lower()
        izvor = (payload.get("izvor", "") or "").lower()
        naziv = (payload.get("naziv_dokumenta", "") or "").lower()
        if "logo" in tekst[:200] or "logo" in izvor or "logo" in naziv:
            return True
        if "fotobaza" in izvor or "fotobaza" in naziv:
            return True
        return False

    # Ako korisnik traži specifično (ruža vetrova, klimatski, mapa)
    specificni_kw = ["vetrova", "ruza vetrova", "wind", "klimatski", "klimadijagram",
                     "temperatura", "padavine", "mapa", "karta"]
    if any(kw in u for kw in specificni_kw):
        filtrirane = []
        for p in points:
            payload = p.payload or {}
            if je_logo(payload):
                continue
            tekst = sredi_upit(payload.get("tekst", "") or "")
            ocr = sredi_upit(payload.get("ocr_tekst", "") or "")
            opis = sredi_upit(payload.get("vizuel_opis", "") or "")
            url = payload.get("Link", "") or payload.get("slika_url", "")
            izvor = payload.get("izvor", "") or "Dijagram"

            # Pretraga po svim tekstualnim poljima
            full_text = f"{ocr} {tekst} {opis} {izvor}"
            if ("vetrova" in full_text or "wind" in full_text
                or "klim" in full_text or "mapa" in full_text or "karta" in full_text):
                if url and url.startswith("http"):
                    caption = opis[:80] if opis else izvor
                    filtrirane.append((url, caption))
        if filtrirane:
            return f"**Pronađeno {len(filtrirane)} dijagrama:**", filtrirane[:6]

    # Filtriraj logo zapise
    slike = []
    for p in points:
        payload = p.payload or {}
        if je_logo(payload):
            continue
        url = payload.get("Link", "") or payload.get("slika_url", "")
        if url and url.startswith("http"):
            opis = payload.get("vizuel_opis", "")
            izvor = payload.get("izvor", "") or "Dijagram"
            caption = opis[:80] if opis else izvor
            slike.append((url, caption))

    if not slike:
        return "⚠️ Nema dijagrama u bazi (samo logo zapisi pronađeni).", []

    # Ako korisnik pita za lokaciju (npr. "dijagram za Crni Vrh"), filtriraj
    lokacije = ["crni vrh", "stig", "vranjaca", "donji pek", "beograd", "kucevo",
                "timoska", "banat", "backa", "srem"]
    target_lok = None
    for lok in lokacije:
        if lok in u:
            target_lok = lok
            break

    if target_lok:
        filtrirane = []
        for url, cap in slike:
            if target_lok in cap.lower() or target_lok in u:
                filtrirane.append((url, cap))
        if filtrirane:
            return f"**Pronađeno {len(filtrirane)} dijagrama za '{target_lok}':**", filtrirane[:6]

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
        # Ako clan_pat nije matchovao ali clan_check jeste, fallback
        if not pogodci and clan_check.search(tekst_norm):
            # Nađi poziciju i uzmi okolni tekst
            for cm in clan_check.finditer(tekst_norm):
                start_orig = max(0, cm.start() - 30)
                end_orig = min(len(tekst), cm.end() + 3500)
                clan_tekst = tekst[start_orig:end_orig].strip()
                if len(clan_tekst) >= 30:
                    pogodci.append({
                        "tekst": clan_tekst,
                        "izvor": payload.get("izvor", "") or payload.get("naziv_dokumenta", ""),
                        "duzina": len(clan_tekst),
                    })
                    break

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

    # Pravni član — PRVO! (pre "ima li" detekcije)
    m = re.search(r'(?:clan|clana|clanom|clanu|clane|cl\.|cln\.)\s*(\d+)', u)
    if m:
        return f"clan_{m.group(1)}"

    # EKSTERNO — pitanja o poznatim ličnostima (pre person routing)
    # "Ko je ministar", "Ko je predsednik" itd. — nisu naši zaposleni
    ekstern_licnosti = [
        "ministar", "predsednik", "premijer", "vladar", "kralj",
        "kraljica", "generalni sekretar", "guverner", "ambasador",
        "predsedavajuci", "potpredsednik", "selo",
    ]
    if any(kw in u for kw in ekstern_licnosti):
        # Samo ako NIJE pitanje o našem direktoru
        if "direktor" not in u and "zamenik" not in u and "biro" not in u:
            return "eksterno"

    # Opšta eksterna pitanja (pre person routing)
    ekstern_opste = [
        "sta je", "sta znaci", "sta predstavlja",
        "kako se zove", "kako se zovu",
        "gde se nalazi", "koliko kosta", "koliko je",
    ]
    if any(kw in u for kw in ekstern_opste):
        # Ali ne za naše ljude
        if "zaposlen" not in u and "biro" not in u and "srbija" not in u and "ministar" not in u and "predsednik" not in u:
            # Specifično za "Ko je" + opšta pitanja
            if "ko je" in u and not any(im in u for im in [" vamovic", " caldovic", " mihajlovic", " bojana", " darko", " arsenije", " aleksandra", " bosko", " malesevic"]):
                return "eksterno"

    # Direktor (pre liste, jer "direktor" sadrži specifičniji pojam)
    if "direktor" in u and "zamenik" not in u:
        return "direktor"

    # Traženje osobe po imenu: "pronađi [Ime]", "nađi [Ime]", "ima li [Ime]"
    if any(kw in u for kw in ["pronadi", "nadji", "nadjem",
                                "ima li", "gde je", "ko je to", "sta je sa",
                                "da li", "dal je", "dal i", "je li",
                                "ko je ", "koja je ", "koje je ",
                                "radi li", "kako se zove", "kako se zovu"]):
        # Ali NE ako je o direktoru ili listi zaposlenih
        if "direktor" not in u and "zaposleni" not in u and "lista" not in u:
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
                elif tip == "eksterno":
                    # Direktno idi na Tavily — ne tražimo u bazi
                    ext = external_search(user_input)
                    if ext:
                        odgovor = f"🌐 **Spoljni izvori:**\n\n{ext}"
                        meta = "\n\n<sub>🌐 Eksterni search (Tavily)</sub>"
                    else:
                        odgovor = "⚠️ Nemam pristup eksternim izvorima (Tavily ključ?)."
                        meta = ""
                    slike = []
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
