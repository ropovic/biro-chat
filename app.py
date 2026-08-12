import os
import streamlit as st

# Postavljanje konfiguracije stranice (obavezno na samom početku)
st.set_page_config(
    page_title="BiroChat AI",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded"
)

from rag_engine import ask_birochat
from r2_sync import sync_employees_from_r2

# Bezbedna sinhronizacija sa R2 samo jednom pri pokretanju aplikacije
if "r2_synced" not in st.session_state:
    try:
        sync_employees_from_r2()
        st.session_state.r2_synced = True
    except Exception as e:
        st.session_state.r2_synced = False
        print(f"R2 sinhronizacija nije izvršena: {e}")

# Inicijalizacija istorije poruka
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Zdravo! Ja sam **BiroChat AI** asistent. Možete me pitati o zaposlenima, rukovodstvu, geodetama, blagajnicima, kao i o ostalim dokumentima i informacijama vezanim za Biro."
        }
    ]

# --- SIDEBAR (BOČNA TRAKA) ---
with st.sidebar:
    st.image("https://raw.githubusercontent.com/streamlit/streamlit/main/docs/static/logo.png", width=120)
    st.title("🏢 BiroChat Admin")
    st.markdown("---")
    
    st.subheader("⚙️ Status sistema")
    if st.session_state.get("r2_synced"):
        st.success("Cloudflare R2: Sinhronizovano")
    else:
        st.warning("Cloudflare R2: Aktivna lokalna/Qdrant baza")

    st.markdown("---")
    if st.button("🗑️ Očisti istoriju razgovora", use_container_width=True):
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "Istorija je očišćena. Kako vam mogu pomoći?"
            }
        ]
        st.rerun()

    st.markdown("---")
    st.caption("BiroChat v2.0 | Powered by Qdrant & Groq / Gemini")

# --- GLAVNI INTERFEJS ---
st.title("🏢 BiroChat AI Asistent")
st.caption("Pretraga interne dokumentacije i profila zaposlenih")

# Prikaz prethodnih poruka iz istorije
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        
        # Ako poruka sadrži kartice zaposlenih ili logotipa
        if "matched_employees" in msg and msg["matched_employees"]:
            st.markdown("---")
            cols = st.columns(min(len(msg["matched_employees"]), 3))
            for idx, emp in enumerate(msg["matched_employees"]):
                col = cols[idx % 3]
                with col:
                    img_src = emp.get("image_url") or emp.get("photo_filename")
                    if img_src:
                        st.image(img_src, caption=emp.get("name", ""), use_container_width=True)
                    st.write(f"**{emp.get('name', '')}**")
                    st.caption(f"💼 {emp.get('role', 'Zaposleni')}")
                    
        # Prikaz izvora ako postoje
        if "sources" in msg and msg["sources"]:
            with st.expander("📚 Korišćeni interni izvori"):
                for src in msg["sources"]:
                    st.write(f"- {src}")
                    
        if "web_sources" in msg and msg["web_sources"]:
            with st.expander("🌐 Korišćeni web izvori"):
                for wsrc in msg["web_sources"]:
                    st.write(f"- {wsrc}")

# --- OBRAĐIVANJE KORISNIČKOG UNOSA ---
if prompt := st.chat_input("Postavite pitanje (npr. 'Ko je zamenik direktora?', 'Ko su blagajnici?')..."):
    
    # Prikaz korisničke poruke
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Odgovor asistenta
    with st.chat_message("assistant"):
        with st.spinner("Pretražujem bazu i generišem odgovor..."):
            res = ask_birochat(prompt)
            
            answer = res["answer"]
            matched_employees = res.get("matched_employees", [])
            sources = res.get("sources", [])
            web_sources = res.get("web_sources", [])
            provider = res.get("provider", "")

            # Prikaz teksta odgovora
            st.markdown(answer)

            # Prikaz kartica sa fotografijama zaposlenih/logotipa ako su pronađeni
            if matched_employees:
                st.markdown("---")
                cols = st.columns(min(len(matched_employees), 3))
                for idx, emp in enumerate(matched_employees):
                    col = cols[idx % 3]
                    with col:
                        img_src = emp.get("image_url") or emp.get("photo_filename")
                        if img_src:
                            st.image(img_src, caption=emp.get("name", ""), use_container_width=True)
                        st.write(f"**{emp.get('name', '')}**")
                        st.caption(f"💼 {emp.get('role', 'Zaposleni')}")

            # Prikaz izvora
            if sources:
                with st.expander("📚 Korišćeni interni izvori"):
                    for src in sources:
                        st.write(f"- {src}")

            if web_sources:
                with st.expander("🌐 Korišćeni web izvori"):
                    for wsrc in web_sources:
                        st.write(f"- {wsrc}")

            if provider:
                st.caption(f"🤖 Model: {provider}")

    # Cuvanje u istoriju
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "matched_employees": matched_employees,
        "sources": sources,
        "web_sources": web_sources,
        "provider": provider
    })