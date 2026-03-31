"""
Health Insurance AI Assistant
Uses Groq (LLaMA 3.3-70b), CrewAI agents, ChromaDB, Tavily, and Streamlit.
Compatible with Python 3.13 and LangChain v0.3+
"""

import os
import tempfile
import streamlit as st
from dotenv import load_dotenv

# ── Load environment variables ──────────────────────────────────────────────
load_dotenv()
GROQ_API_KEY   = os.getenv("GROQ_API_KEY", "")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")

# ── LangChain imports (v0.3 compatible) ──────────────────────────────────────
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# ── CrewAI imports ────────────────────────────────────────────────────────────
from crewai import Agent, Task, Crew, Process

# ════════════════════════════════════════════════════════════════════════════
# 1.  Core LLM  – ChatGroq with llama-3.3-70b-versatile
# ════════════════════════════════════════════════════════════════════════════
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=GROQ_API_KEY,
    temperature=0.3,
)


# ════════════════════════════════════════════════════════════════════════════
# 2.  Tool – hospital_search  (uses TavilySearchResults with k=3)
# ════════════════════════════════════════════════════════════════════════════
@tool("hospital_search")
def hospital_search(location: str) -> str:
    """
    Search for hospitals near a given city or location using Tavily.
    Returns a formatted list of nearby hospitals with details.
    """
    try:
        searcher = TavilySearchResults(
            k=3,
            tavily_api_key=TAVILY_API_KEY,
        )
        query   = f"best hospitals near {location} with emergency services"
        results = searcher.invoke(query)

        if not results:
            return f"{location} जवळ कोणतेही रुग्णालय सापडले नाही."

        lines = []
        for i, r in enumerate(results, 1):
            title   = r.get("title", "अज्ञात रुग्णालय")
            url     = r.get("url", "")
            snippet = r.get("content", "")[:200]
            lines.append(f"{i}. **{title}**\n   {snippet}\n   🔗 {url}")

        return "\n\n".join(lines)

    except Exception as exc:
        return f"रुग्णालय शोधताना त्रुटी: {exc}"


# ════════════════════════════════════════════════════════════════════════════
# 3.  Policy PDF helper  – ChromaDB + PyPDFLoader
# ════════════════════════════════════════════════════════════════════════════
def build_policy_context(pdf_path: str) -> str:
    """
    Load a PDF with PyPDFLoader, embed pages into ChromaDB (in-memory),
    and return the full text content (truncated) for the policy agent.
    """
    try:
        loader = PyPDFLoader(pdf_path)
        pages  = loader.load()

        if not pages:
            return "PDF मध्ये कोणताही मजकूर आढळला नाही."

        # Store in ChromaDB (in-memory collection for this session)
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vectorstore = Chroma.from_documents(
            documents=pages,
            embedding=embeddings,
            collection_name="policy_docs",
        )

        # Return raw text (first 6 000 chars) for agent context
        full_text = "\n".join(p.page_content for p in pages)
        return full_text[:6000]

    except Exception as exc:
        return f"PDF वाचताना त्रुटी: {exc}"


# ════════════════════════════════════════════════════════════════════════════
# 4.  CrewAI agent & crew factory
# ════════════════════════════════════════════════════════════════════════════
def create_crew(symptoms: str, location: str, policy_text: str | None) -> Crew:

    # ── Agent 1 : Symptom Analyzer ──────────────────────────────────────────
    symptom_analyzer = Agent(
        role="वैद्यकीय लक्षण विश्लेषक",
        goal=(
            "रुग्णाची लक्षणे ऐकून योग्य वैद्यकीय तज्ञाचे नाव सुचवा "
            "आणि प्राथमिक निदान मराठीत द्या."
        ),
        backstory=(
            "तुम्ही एक अनुभवी वैद्यकीय सल्लागार आहात जे नेहमी मराठीत "
            "उत्तर देतात. लक्षणांच्या आधारे कोणत्या विशेषज्ञाकडे जावे "
            "हे ओळखणे तुमची खासियत आहे."
        ),
        llm=llm,
        verbose=False,
        allow_delegation=False,
    )

    # ── Agent 2 : Policy Expert ─────────────────────────────────────────────
    if policy_text:
        policy_backstory = (
            "तुम्ही एक विमा पॉलिसी तज्ञ आहात. "
            "खालील पॉलिसी माहितीच्या आधारेच उत्तर द्या:\n\n"
            f"{policy_text}\n\n"
            "पॉलिसीमध्ये नमूद नसलेल्या गोष्टींबद्दल "
            "'या पॉलिसीत नमूद नाही' असे सांगा."
        )
    else:
        policy_backstory = (
            "तुम्ही एक विमा पॉलिसी तज्ञ आहात. "
            "सध्या कोणतीही पॉलिसी PDF उपलब्ध नाही."
        )

    policy_expert = Agent(
        role="विमा पॉलिसी तज्ञ",
        goal=(
            "अपलोड केलेल्या विमा पॉलिसीच्या आधारे कव्हरेज माहिती मराठीत द्या. "
            "PDF नसल्यास 'कृपया पॉलिसी PDF अपलोड करा.' असे म्हणा."
        ),
        backstory=policy_backstory,
        llm=llm,
        verbose=False,
        allow_delegation=False,
    )

    # ── Agent 3 : Location Expert ───────────────────────────────────────────
    location_expert = Agent(
        role="स्थान तज्ञ",
        goal=f"{location} जवळील रुग्णालये शोधा आणि मराठीत यादी द्या.",
        backstory=(
            "तुम्ही एक स्थानिक आरोग्य सेवा मार्गदर्शक आहात जे "
            "hospital_search टूल वापरून नेहमी मराठीत जवळच्या "
            "रुग्णालयांची माहिती देतात."
        ),
        llm=llm,
        tools=[hospital_search],
        verbose=False,
        allow_delegation=False,
    )

    # ── Tasks ────────────────────────────────────────────────────────────────
    task_symptoms = Task(
        description=(
            f"रुग्णाची लक्षणे: {symptoms}\n\n"
            "या लक्षणांचे विश्लेषण करा. संभाव्य आजार, "
            "कोणत्या तज्ञ डॉक्टरकडे जावे, आणि प्राथमिक सल्ला "
            "मराठीत द्या. इंग्रजी वापरू नका."
        ),
        expected_output=(
            "मराठीत: संभाव्य आजार, सुचवलेले तज्ञ, आणि प्राथमिक सल्ला."
        ),
        agent=symptom_analyzer,
    )

    task_policy = Task(
        description=(
            (
                f"रुग्णाची लक्षणे: {symptoms}\n\n"
                "वरील पॉलिसी माहितीच्या आधारे या आजारासाठी विमा कव्हरेज "
                "आहे का ते मराठीत सांगा. रुग्णालयात भरती, औषधे, "
                "शस्त्रक्रिया यांचा समावेश आहे का ते सांगा."
            )
            if policy_text
            else "कोणतीही पॉलिसी PDF उपलब्ध नाही. योग्य संदेश द्या."
        ),
        expected_output=(
            "मराठीत: विमा कव्हरेज तपशील किंवा PDF अपलोड विनंती."
        ),
        agent=policy_expert,
    )

    task_location = Task(
        description=(
            f"'{location}' या ठिकाणाजवळील रुग्णालये शोधा. "
            "hospital_search टूल वापरा आणि किमान ३ रुग्णालयांची "
            "नावे व माहिती मराठीत द्या."
        ),
        expected_output=(
            "मराठीत: जवळच्या रुग्णालयांची यादी नाव व माहितीसह."
        ),
        agent=location_expert,
    )

    return Crew(
        agents=[symptom_analyzer, policy_expert, location_expert],
        tasks=[task_symptoms, task_policy, task_location],
        process=Process.sequential,
        verbose=False,
    )


# ════════════════════════════════════════════════════════════════════════════
# 5.  Streamlit UI
# ════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="आरोग्य विमा सहाय्यक",
    page_icon="🏥",
    layout="wide",
)

st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem; font-weight: 700;
        color: #1a6b3c; text-align: center;
        padding: 0.4rem 0 0.8rem;
    }
    .section-card {
        background: #f0faf4;
        border-left: 5px solid #1a6b3c;
        border-radius: 8px;
        padding: 1.2rem 1.5rem;
        margin-bottom: 1.2rem;
    }
    .section-title {
        font-size: 1.05rem; font-weight: 700;
        color: #1a6b3c; margin-bottom: 0.5rem;
    }
    .stButton > button {
        background-color: #1a6b3c; color: white;
        border-radius: 8px; border: none;
        font-size: 1rem; width: 100%;
    }
    .stButton > button:hover { background-color: #145c32; }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown(
    '<div class="main-title">🏥 आरोग्य विमा AI सहाय्यक</div>',
    unsafe_allow_html=True,
)
st.markdown(
    "<p style='text-align:center;color:#555;'>"
    "लक्षणे सांगा · पॉलिसी अपलोड करा · जवळचे रुग्णालय शोधा"
    "</p>",
    unsafe_allow_html=True,
)
st.divider()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("📄 विमा पॉलिसी PDF")
    st.caption("तुमची आरोग्य विमा पॉलिसी येथे अपलोड करा")

    uploaded_pdf = st.file_uploader(
        "PDF अपलोड करा",
        type=["pdf"],
        help="फक्त PDF फाइल स्वीकारली जाते",
    )

    if uploaded_pdf:
        st.success(f"✅ अपलोड झाले: {uploaded_pdf.name}")

    st.divider()
    st.markdown("**API कनेक्शन स्थिती**")
    st.markdown(f"{'✅' if GROQ_API_KEY else '❌'} Groq API Key")
    st.markdown(f"{'✅' if TAVILY_API_KEY else '❌'} Tavily API Key")
    st.divider()
    st.caption(
        "⚠️ हे साधन केवळ माहितीसाठी आहे. "
        "कृपया निदानासाठी नेहमी डॉक्टरांचा सल्ला घ्या."
    )

# ── Main inputs ───────────────────────────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    symptoms_input = st.text_area(
        "🤒 लक्षणे (Symptoms)",
        placeholder="उदा. ताप, खोकला, छातीत दुखणे, श्वास घेण्यास त्रास...",
        height=150,
    )

with col2:
    location_input = st.text_input(
        "📍 शहर / ठिकाण (City / Location)",
        placeholder="उदा. पुणे, मुंबई, नागपूर...",
    )

run_btn = st.button("🔍 अहवाल तयार करा", use_container_width=True)

# ── Run ───────────────────────────────────────────────────────────────────────
if run_btn:
    if not symptoms_input.strip():
        st.warning("⚠️ कृपया लक्षणे प्रविष्ट करा.")
        st.stop()
    if not location_input.strip():
        st.warning("⚠️ कृपया शहर किंवा ठिकाण प्रविष्ट करा.")
        st.stop()
    if not GROQ_API_KEY:
        st.error("❌ GROQ_API_KEY .env फाइलमध्ये सेट केलेली नाही.")
        st.stop()

    # ── Process PDF ───────────────────────────────────────────────────────
    policy_text = None
    if uploaded_pdf is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_pdf.read())
            tmp_path = tmp.name
        with st.spinner("📄 PDF वाचत आहे..."):
            policy_text = build_policy_context(tmp_path)
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    # ── Run CrewAI ────────────────────────────────────────────────────────
    with st.spinner("🤖 AI एजंट्स विश्लेषण करत आहेत… कृपया प्रतीक्षा करा (30-60 सेकंद)…"):
        try:
            crew   = create_crew(symptoms_input, location_input, policy_text)
            result = crew.kickoff()

            # Extract per-task outputs
            task_outputs = [str(t.output) if t.output else "" for t in crew.tasks]
            diagnosis_text = task_outputs[0] if len(task_outputs) > 0 else str(result)
            insurance_text = task_outputs[1] if len(task_outputs) > 1 else "माहिती उपलब्ध नाही."
            hospitals_text = task_outputs[2] if len(task_outputs) > 2 else "माहिती उपलब्ध नाही."

        except Exception as exc:
            st.error(f"❌ त्रुटी आली: {exc}")
            st.exception(exc)
            st.stop()

    # ── Combined Report ───────────────────────────────────────────────────
    st.divider()
    st.subheader("📋 एकत्रित आरोग्य अहवाल (मराठी)")

    for emoji, num, title, content in [
        ("🩺", "१", "निदान व तज्ञ सल्ला",    diagnosis_text),
        ("📑", "२", "विमा कव्हरेज",           insurance_text),
        ("🏥", "३", "जवळची रुग्णालये",         hospitals_text),
    ]:
        st.markdown(
            f'<div class="section-card">'
            f'<div class="section-title">{emoji} {num}. {title}</div>'
            f'{content}'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.success("✅ अहवाल तयार झाला!")
