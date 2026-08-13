import streamlit as st
from rag_engine import ask_birochat

# Podešavanje stranice
st.set_page_config(
    page_title="BiroChat AI",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inicijalizacija istorije poruka u session state-u
if "messages" not in st.session_state:
    st.session_state.messages = []


def render_employee_cards(employees: list):
    """Renderuje vizuelne kartice za zaposlene sa slikama sa URL-a."""
    if not employees:
        return

    st.markdown("### 👤 Profil / Rukovodstvo")
    cols = st.columns(min(len(employees), 3))
    
    for idx, emp in enumerate(employees):
        col = cols[idx % 3]
        with col:
            with st.container(border=True):
                img_url = (
                    emp.get("image_url") or 
                    emp.get("photo_url") or 
                    emp.get("image") or 
                    emp.get("slika")
                )
                
                if img_url and isinstance(img_url, str) and img_url.startswith(("http://", "https://")):
                    try:
                        st.image(
                            img_url,
                            caption=emp.get("name", ""),
                            use_container_width=True
                        )
                    except Exception:
                        st.warning("⚠️ Slika ne može da se učita")
                else:
                    st.info("👤 Nema priložene fotografije")

                name = emp.get("name", "Nepoznati zaposleni")
                role = emp.get("role", "Funkcija nije navedena")
                
                st.subheader(name)
                st.caption(f"💼 **Funkcija:** {role}")


# Zaglavlje interfejsa
st.title("🏢 BiroChat Interni Asistent")
st.caption("Pretraga interne dokumentacije, ugovora, POGŠ, opreme i profila zaposlenih.")

# 🧪 Dugmad za brzo testiranje baze
st.markdown("##### 🧪 Brza test pitanja:")
btn_col1, btn_col2, btn_col3 = st.columns(3)
btn_col4, btn_col5, btn_col6 = st.columns(3)

triggered_query = None

if btn_col1.button("1. Ko je direktor Biroa?", use_container_width=True):
    triggered_query = "Ko je direktor Biroa?"
if btn_col2.button("2. Ko su zamenici direktora?", use_container_width=True):
    triggered_query = "Ko su zamenici direktora?"
if btn_col3.button("3. Postojeće POGŠ u bazi?", use_container_width=True):
    triggered_query = "Koje osnove gazdovanja šumama (POGŠ) postoje u bazi."
if btn_col4.button("4. Štampači u Birou?", use_container_width=True):
    triggered_query = "Koji štampači se koriste u Birou?"
if btn_col5.button("5. Potrebni toneri?", use_container_width=True):
    triggered_query = "Koji toneri su potrebni za štampače?"
if btn_col6.button("6. Član 14 ugovora?", use_container_width=True):
    triggered_query = "Navedi član 14 kolektivnog ugovora."

# Polje za slobodan unos
chat_input = st.chat_input("Postavite pitanje o dokumentima ili zaposlenima...")

# Preuzimanje unosa (sa dugmeta ili iz polja)
user_input = triggered_query or chat_input

# Prikaz istorije poruka
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("matched_employees"):
            render_employee_cards(msg["matched_employees"])
        if msg.get("sources"):
            with st.expander("📚 Korišćeni izvori iz baze"):
                for src in msg["sources"]:
                    st.write(f"- {src}")

# Obrada novog pitanja
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Pretražujem bazu podataka..."):
            res = ask_birochat(user_input)
            
            answer = res.get("answer", "")
            matched_employees = res.get("matched_employees", [])
            sources = res.get("sources", [])

            st.markdown(answer)

            if matched_employees:
                render_employee_cards(matched_employees)

            if sources:
                with st.expander("📚 Korišćeni izvori iz baze"):
                    for src in sources:
                        st.write(f"- {src}")

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "matched_employees": matched_employees,
        "sources": sources
    })
    st.rerun()