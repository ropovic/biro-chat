import streamlit as st
from rag_engine import query_biro_system
from r2_sync import sync_employees_from_r2

st.set_page_config(page_title="BiroChat - Upravljanje i Zaposleni", page_icon="🏢", layout="wide")

st.title("🏢 BiroChat - Interni Asistent")
st.markdown("Pretraga baze dokumenata, organizacione strukture i zaposlenih u Birou.")

# Bočna traka za administraciju
with st.sidebar:
    st.header("⚙️ Administracija")
    if st.button("🔄 Osveži zaposlene iz Cloudflare R2"):
        with st.spinner("Preuzimanje slika i opisa..."):
            sync_employees_from_r2()
            st.success("Baza zaposlenih je uspešno ažurirana!")

# Istorija poruka u sesiji
if "messages" not in st.session_state:
    st.session_state.messages = []

# Prikaz prethodnih poruka
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "employees" in msg and msg["employees"]:
            st.write("---")
            st.subheader("🖼️ Detektovani profili zaposlenih:")
            cols = st.columns(min(len(msg["employees"]), 3))
            for idx, emp in enumerate(msg["employees"]):
                with cols[idx % 3]:
                    st.image(emp["image_url"], use_column_width=True)
                    badge = "⭐ DIREKTOR" if emp.get("is_director") else "👤 ZAPOSLENI"
                    st.markdown(f"**{emp['name']}**\n\n`{badge}`")
                    st.caption(f"**Rola:** {emp['role']}")
                    if emp.get("description"):
                        st.write(emp["description"])

# Korisnički unos
if prompt := st.chat_input("Postavite pitanje (npr. 'Ko je direktor Biroa?' ili 'Prikaži zaposlene')..."):
    # Prikaz korisničkog pitanja
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generisanje odgovora
    with st.chat_message("assistant"):
        with st.spinner("Pretražujem bazu dokumenata i Cloudflare R2..."):
            answer, employees = query_biro_system(prompt)
            
            st.markdown(answer)
            
            # Prikaz vizuelnih kartica zaposlenih ako ih ima u rezultatima
            if employees:
                st.write("---")
                st.subheader("🖼️ Identifikovani profili iz baze:")
                cols = st.columns(min(len(employees), 3))
                for idx, emp in enumerate(employees):
                    with cols[idx % 3]:
                        st.image(emp["image_url"], caption=emp["name"], use_column_width=True)
                        badge = "⭐ DIREKTOR BIROA" if emp.get("is_director") else "👤 ZAPOSLENI"
                        st.markdown(f"**{emp['name']}**")
                        st.markdown(f"**Status:** `{badge}`")
                        st.markdown(f"**Pozicija:** {emp['role']}")
                        if emp.get("description"):
                            st.caption(emp["description"])

            # Sačuvanje u sesiju
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "employees": employees
            })