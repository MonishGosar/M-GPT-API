import os
import pickle
import numpy as np
import faiss
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY is not set")

genai.configure(api_key=GEMINI_API_KEY)

# Load chunks, embeddings, and FAISS index
with open("chunks.pkl", "rb") as f:
    chunks = pickle.load(f)
embeddings = np.load("embeddings.npy")
index = faiss.read_index("faiss_index.bin")

# Initialize FastAPI
app = FastAPI(title="M-GPT API with Gemini", version="1.0.0")

class Query(BaseModel):
    message: str

# Optional: Use lightweight model for query embedding generation
from sentence_transformers import SentenceTransformer
query_model = SentenceTransformer('paraphrase-MiniLM-L3-v2')  # Light model for query only

def generate_response_with_gemini(query, context):
    prompt = f"""
    SYSTEM:
    You are Monish Gosar, 22-year-old working in Data Science and ML.
    - Start with "Hi, I'm Monish," but only in the first message.
    - Be short, crisp, kind, and human-sounding.
    - Answer grounded in the CONTEXT below.

    CONTEXT:
    {context}

    USER:
    {query}
    """
    gemini_model = genai.GenerativeModel('gemini-2.5-flash')
    response = gemini_model.generate_content(prompt)
    return response.text

@app.post("/chat")
async def chat(query: Query):
    if not query.message.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    # Generate embedding for the user query
    query_embedding = query_model.encode([query.message])
    D, I = index.search(query_embedding, 3)  # Top 3 similar chunks
    context = "\n".join([chunks[i] for i in I[0]])

    # Generate Gemini response using the retrieved context
    response = generate_response_with_gemini(query.message, context)
    return {"response": response}
