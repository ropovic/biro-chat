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
    
    # Prikaz u mrežastom rasporedu (do 3 kartice u redu)
    cols = st.columns(min(len(employees), 3))
    
    for idx, emp in enumerate(employees):
        col = cols[idx % 3]
        with col:
            with st.container(border=True):
                # Ekstrakcija URL-a slike iz različitih mogućih ključeva u bazi
                img_url = (
                    emp.get("image_url") or 
                    emp.get("photo_url") or 
                    emp.get("image") or 
                    emp.get("slika")
                )
                
                # Učitavanje slike preko st.image() ako je URL validan
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

                # Osnovni podaci o zaposlenom
                name = emp.get("name", "Nepoznati zaposleni")
                role = emp.get("role", "Funkcija nije navedena")
                
                st.subheader(name)
                st.caption(f"💼 **Funkcija:** {role}")


# Glavni interfejs
st.title("🏢 BiroChat Interni Asistent")
st.caption("Pretraga interne dokumentacije, ugovora i profila zaposlenih.")

# Prikaz prethodnih poruka iz istorije
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        
        # Ako poruka sadrži profil zaposlenog, prikaži karticu
        if msg.get("matched_employees"):
            render_employee_cards(msg["matched_employees"])
            
        # Prikaz izvora ako postoje
        if msg.get("sources"):
            with st.expander("📚 Korišćeni izvori iz baze"):
                for src in msg["sources"]:
                    st.write(f"- {src}")

# Unos pitanja
if user_input := st.chat_input("Postavite pitanje o dokumentima ili zaposlenima..."):
    # 1. Prikaz korisničkog pitanja
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # 2. Obrada odgovora preko RAG sistema
    with st.chat_message("assistant"):
        with st.spinner("Pretražujem bazu podataka..."):
            res = ask_birochat(user_input)
            
            answer = res.get("answer", "")
            matched_employees = res.get("matched_employees", [])
            sources = res.get("sources", [])

            # Ispis tekstualnog odgovora
            st.markdown(answer)

            # Prikaz kartica sa slikom ako je prepoznat zaposleni
            if matched_employees:
                render_employee_cards(matched_employees)

            # Prikaz izvora
            if sources:
                with st.expander("📚 Korišćeni izvori iz baze"):
                    for src in sources:
                        st.write(f"- {src}")

    # 3. Čuvanje kompletnog odgovora u istoriji radi očuvanja stanja slika pri osvežavanju
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "matched_employees": matched_employees,
        "sources": sources
    })