# 🏥 आरोग्य विमा AI सहाय्यक 
### Health Insurance AI Assistant (Marathi RAG)

A fully Marathi-language AI assistant that analyzes symptoms, checks insurance coverage from an uploaded PDF, and finds nearby hospitals — all powered by **Groq LLaMA 3.3-70b**, **CrewAI**, **ChromaDB**, **Tavily**, and **Streamlit**.

---

## 🏗️ Architecture

- **UI:** Streamlit (Marathi Interface)
- **Orchestration:** CrewAI (Multi-Agent System)
- **LLM:** Llama-3.3-70b via Groq (Fast Inference)
- **Knowledge Base:** ChromaDB (Vector Store for PDF RAG)
- **Search Tool:** Tavily API (Real-time Hospital Search)

---

## 🤖 Agents & Roles

| Agent | Role (Marathi) | Task |
|-------|------|------|
| **Symptom Analyzer** | वैद्यकीय लक्षण विश्लेषक | Analyzes symptoms & suggests specialist |
| **Policy Expert** | विमा धोरण तज्ञ | Answers queries from uploaded PDF |
| **Location Expert** | स्थान विशेषज्ञ | Finds nearby hospitals via Tavily |

---

## 🚀 Setup & Installation

1. **Clone the Repo:**
   `git clone https://github.com/तुमचे-युजरनेम/Marathi-Health-Insurance-AI.git`

2. **Install Dependencies:**
   `pip install -r requirements.txt`

3. **Configure Environment:**
   Create a `.env` file and add your `GROQ_API_KEY` and `TAVILY_API_KEY`.

4. **Run App:**
   `streamlit run app.py`

---

## 📋 GSoC 2026 Note (Kubeflow / Mifos)
This project serves as a **Proof of Concept (PoC)** for localized AI solutions. It demonstrates:
- Multi-agent coordination using CrewAI.
- Efficient RAG pipeline with ChromaDB.
- Marathi Language support for financial and health inclusion.

---

## ⚠️ Disclaimer
This is an AI-powered tool for informational purposes. Always consult professional doctors and insurance advisors for official advice.
