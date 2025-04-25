import os
import pickle
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import google.generativeai as genai
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from fastapi import Request
import logging
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure Gemini API - Use environment variable for API key
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable is not set")

genai.configure(api_key=GEMINI_API_KEY)

def load_and_split_documents():
    """Load about_me.txt and split into chunks."""
    # Load text file
    file_path = os.path.join(os.path.dirname(__file__), "About-Me-2.txt")
    with open(file_path, "r", encoding="utf-8") as f:
        txt_text = f.read()

    # Split into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)
    chunks = text_splitter.split_text(txt_text)
    return chunks

def save_chunks(chunks):
    """Save document chunks to a pickle file."""
    with open("chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)

def load_chunks():
    """Load document chunks from a pickle file."""
    with open("chunks.pkl", "rb") as f:
        return pickle.load(f)

def generate_embeddings(chunks, model):
    """Generate embeddings for document chunks using SentenceTransformer."""
    embeddings = model.encode(chunks)
    return embeddings

def create_or_load_faiss_index(embeddings):
    """Create a new FAISS index or load an existing one."""
    if os.path.exists("faiss_index.bin"):
        index = faiss.read_index("faiss_index.bin")
    else:
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings)
        faiss.write_index(index, "faiss_index.bin")
    return index

# Initialize model and load data
model = SentenceTransformer('all-MiniLM-L6-v2')

if os.path.exists("chunks.pkl"):
    chunks = load_chunks()
else:
    chunks = load_and_split_documents()
    save_chunks(chunks)


embeddings = generate_embeddings(chunks, model)
index = create_or_load_faiss_index(embeddings)

# FastAPI app
app = FastAPI(title="M-GPT API", version="1.0.0")

# Add CORS middleware with more specific configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Query(BaseModel):
    message: str

def generate_response(query, context):
    """Generate a response using Gemini 1.5 Flash with the given context."""
    prompt = f"""
        SYSTEM:
        You are Monish Gosar, 22 year old working in the field of Data Science and Machine Learning.
        When replying:
        - Always start with "Hi, I'm Monish," but only in the first message. 
        - Keep it short, crisp, and human-sounding. 
        - Always be kind and warm  
        - Ground every answer in the provided context , keep it short and concise.
        - You are the best conversationalist, so you can answer any question.
        
        CONTEXT: {context}
        USER: {query}
"""
    gemini_model = genai.GenerativeModel('gemini-2.5-flash')
    response = gemini_model.generate_content(prompt)
    return response.text

# @app.post("/api/chat")
# async def chat(query: Query):
#     """POST endpoint for chat queries."""
#     if not query.message.strip():
#         raise HTTPException(status_code=400, detail="Query cannot be empty")
    
#     query_embedding = model.encode([query.message])
#     D, I = index.search(query_embedding, 3)  # Retrieve top 3 chunks
#     context = "\n".join([chunks[i] for i in I[0]])
#     response = generate_response(query.message, context)
#     return {"response": response}

# For backward compatibility
@app.post("/chat")
async def chat_post(request: Request):
    try:
        data = await request.json()
        user_query = data.get("message", "").strip()

        if not user_query:
            raise HTTPException(status_code=400, detail="Query cannot be empty")

        query_embedding = model.encode([user_query])
        D, I = index.search(query_embedding, 3)
        context = "\n".join([chunks[i] for i in I[0]])
        response = generate_response(user_query, context)
        return {"response": response}
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")
