import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import google.generativeai as genai

# ───────────────────────────────────────────────────────────────
# 1. Environment & Gemini setup
# ───────────────────────────────────────────────────────────────
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY environment variable is missing")

genai.configure(api_key=GEMINI_API_KEY)

# Instantiate the model once and re-use it
# (Rename to the exact model you have access to, e.g. "models/gemini-1.5-flash")
GEMINI_MODEL = genai.GenerativeModel("gemini-2.0-flash")

# ───────────────────────────────────────────────────────────────
# 2. Prompt helpers
# ───────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """
You are MonishGPT, a friendly first-person digital version of Monish.

Reply rules:
- Always start with “Hi, I'm Monish,”
- Keep answers short, crisp, human-sounding, kind and warm
- Ground every answer in the provided context
- Invite users to learn more about me and explore the site.
- I am a data scientist working in credit risk analytics and AI.

{CONTEXT}
""".strip()

# (Truncated here for brevity — keep your full résumé / context block)
CONTEXT_BLOCK = """## 1. Professional Profile

**Monish Gosar**  
I am a data scientist working in credit risk analytics and AI.

monish.emailbox@gmail.com | +91 7045636928 | [LinkedIn](https://linkedin.com/in/monish-gosar) | [GitHub](https://github.com/MonishGosar)

### 1.1 Education
- **B.Tech in Data Science**, NMIMS University, Mumbai (Sept 2021 – May 2025)  
  CGPA: 3.46 / 4  
  **Languages & Tools:** Python, SQL, Git, PowerBI, Tableau, AWS, Azure Cloud  
  **Frameworks:** Pandas, NumPy, TensorFlow, Keras, Scikit‑Learn, OpenCV  
  **Databases:** MySQL, PostgreSQL, SQLite

### 1.2 Professional Experience

#### Data Science Intern, Indilabs.ai (Nov 2024 – Ongoing)
- **Risk Analytics & Monitoring:** Developed and deployed a risk intelligence platform for major banks. Engineered portfolio and performance dashboards to automate real‑time monitoring, reducing manual reporting overhead and slashing operational costs.
- **Diagnostics & Forecasting:** Implemented automated diagnostics with behavioral scoring (14% uplift in prediction accuracy) and vintage analysis models to forecast recovery for a \$1 billion lending portfolio.
- **AI Settlement Assistant (IndiBot):** Built an AI agent using Streamlit, Azure AI Studio, GPT‑4o Mini, custom prompt pipelines, PostgreSQL/Cosmos DB, and a three‑component architecture (SQL, Analysis, Visualization agents). Integrated domain knowledge and a two‑level response framework for complex query handling.

#### Python Developer Intern, RE Journal (May 2024 – Aug 2024)
- Applied K‑Means clustering on 1,000+ user profiles, achieving 85% accuracy in segment identification.
- Scraped 150,000+ real estate records with Selenium & Beautiful Soup; designed a scalable PostgreSQL schema.
- Built interactive analytics dashboards in Power BI and Streamlit to surface buyer behavior insights.

#### Data Science Intern, Quantum Software (May 2023 – July 2023)
- Optimized 2G/4G KPI analysis pipelines; improved forecast accuracy using LSTM, ARIMA, and SARIMA models.
- Leveraged AI4Bharat’s Indic language models for 99.3%‑accurate multilingual sentiment analysis.

### 1.3 Projects

- **Contractify (Oct 2024):** RAG‑powered legal contract QA bot with FAISS/Nomic embeddings, Llama3 & Gemini integration, LangChain Q&A pipeline, and Streamlit UI.
- **Audio Classification – Industrial Steel (Sept 2024):** 1D/2D CNNs on FFT spectrograms; 98% F1 score; end‑to‑end ML pipeline on Azure.
- **Money Mule Detection (Feb 2024):** Stacked ensemble (RF, XGBoost, CatBoost) for fraud detection; 0.987 F1; explainability via SHAP & LIME.

### 1.4 Achievements & Leadership
- 2nd Place, LexisNexis Risk Solutions Hackathon
- Top 125, Amazon ML Challenge 2024
- Finalist, Convolve Epoch 2 IDFC Hackathon
- Sub‑Head of R&D, Analytika Data Science Club (led 5‑member team; organized 3 ML/AI events)

---

# Extended Indilabs work in detail :
## Risk Analytics Knowledge Base

### 2.1 Modules Overview

1. **Monitoring Dashboard**
   - **Purpose:** Real‑time portfolio oversight and early risk detection.
   - **Components:** Portfolio, Performance, Distribution & Vintage, Agency dashboards.
   - **Benefits & Cost Reduction:** Automated reporting; proactive interventions; efficient resource allocation; agency performance management.

2. **Prediction & Diagnostics Module**
   1. **Data Preparation & Modeling:** Cleaning, EDA, encoding, baseline model, IV‑based feature selection, KS/ROC‑AUC/F1 evaluation, hyperparameter tuning.
   2. **Decision Tree & Segmentation:** SHAP‑driven top drivers, node‑ID segments (High/Medium/Low risk).
   3. **Stitching & Marginal Curves:** Cumulative vs. incremental performance curves per segment.
   4. **Forecasting:** Projected performance via marginal × allocation; multi‑segment aggregation.
   5. **Diagnostics & Savings:** Targeted strategies, proactive risk management, reduced manual review.

3. **Strategy & Treatments**
   - Segment‑specific treatment rules (e.g., settlement offers, payment plans).
   - Continuous feedback loop to refine negotiation tactics.


## AI Agent Documentation: _IndiBot – Recovery Analytics Assistant_

### 3.1 Overview
- Conversational AI for recovery analytics built on Streamlit, Azure OpenAI, PostgreSQL/Cosmos DB.
- Continuously updated knowledge (as of Apr 02, 2025).

### 3.2 Workflow & Agents
1. **Routing Agent** (`match_question`): predefined vs. free‑form queries
2. **Analysis Agent:** Retrieves reference data or runs `analyze_results`
3. **Query Generator Agent:** `generate_sql_query` for dynamic SQL
4. **Compute Agent:** Executes SQL via `pd.read_sql_query`
5. **Supervisor Agent:** Monitors flow, routes success/errors
6. **Error Agent:** Auto‑diagnoses and retries
7. **Final Analysis:** Formats concise, metric‑driven responses

### 3.3 Future Enhancements
- Explicit `supervise_execution` wrapper
- Intelligent SQL error correction
- Result caching, richer visualizations, stricter domain validation

---

## Personal Profile
- **Hobbies:** Formula 1 engineering deep dives; gaming (RDR 2, Assassin’s Creed, WoW)

"""

def build_prompt(user_query: str) -> str:
    """Combine system prompt, context, and the user’s query."""
    return (
        SYSTEM_PROMPT.format(CONTEXT=CONTEXT_BLOCK)
        + f"\n\nUSER: {user_query}"
    )

def generate_response(user_query: str) -> str:
    prompt = build_prompt(user_query)
    response = GEMINI_MODEL.generate_content(prompt)
    # Strip extra whitespace just in case
    return response.text.strip()


# ───────────────────────────────────────────────────────────────
# 3. FastAPI application
# ───────────────────────────────────────────────────────────────
app = FastAPI(title="MonishGPT API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # 🔐  tighten in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Query(BaseModel):
    message: str

@app.get("/")
def read_root():
    return {"status": "MonishGPT API is live"}

@app.post("/chat")
async def chat(query: Query):
    message = query.message.strip()
    if not message:
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    try:
        answer = generate_response(message)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return {"response": answer}
