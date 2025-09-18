from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import google.generativeai as genai
from typing import Optional, List, Mapping
from pydantic import BaseModel, PrivateAttr
from langchain.llms.base import LLM
import os
from dotenv import load_dotenv
import google.generativeai as genai

# Load variables from .env file
load_dotenv()
# -------------------------------
# Load summaries and compute embeddings
# -------------------------------
with open(r"C:\Users\shruti shreya\Downloads\assessment-infraintelai\task2\summaries.json", "r") as f:
    summaries = json.load(f)

documents = []
metadata = []
for filename, content in summaries.items():
    raw_json = content["raw_output"].strip("``````")
    documents.append(raw_json)
    metadata.append({"filename": filename})

embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
doc_embeddings = embed_model.encode(documents, convert_to_numpy=True)

# -------------------------------
# Configure Gemini LLM
# -------------------------------


# Get API key from environment
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("⚠️ GEMINI_API_KEY not found in environment variables!")

# Configure Gemini
genai.configure(api_key=api_key)  # replace with your key
gemini_model = genai.GenerativeModel("gemini-2.0-flash")

class GeminiLLM(LLM, BaseModel):
    _model: object = PrivateAttr()

    def __init__(self, model, **kwargs):
        super().__init__(**kwargs)
        self._model = model

    @property
    def _llm_type(self) -> str:
        return "gemini-2.0-flash"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        response = self._model.generate_content(prompt)
        return response.text

    @property
    def _identifying_params(self) -> Mapping[str, str]:
        return {"model": self._llm_type}

gemini_llm = GeminiLLM(model=gemini_model)

# -------------------------------
# Simple retrieval function
# -------------------------------
def retrieve(query, top_k=3):
    query_emb = embed_model.encode([query], convert_to_numpy=True)
    similarities = cosine_similarity(query_emb, doc_embeddings)[0]
    top_indices = np.argsort(similarities)[::-1][:top_k]
    results = [{"metadata": metadata[i], "content": documents[i], "score": similarities[i]} for i in top_indices]
    return results

# -------------------------------
# FastAPI App
# -------------------------------
app = FastAPI(title="Medical Records Chatbot")

# Serve static files (CSS)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Set up Jinja2 templates
templates = Jinja2Templates(directory="templates")  # Assuming templates/ folder with chat.html

# Home route
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("chat.html", {"request": request, "answer": None,})

# Chat route
@app.post("/chat", response_class=HTMLResponse)
async def chat(request: Request, query: str = Form(...)):
    top_docs = retrieve(query, top_k=3)
    context_text = "\n\n".join([doc["content"] for doc in top_docs])
    prompt = f"""
    You are a medical assistant. Based on the following patient records, answer the question concisely in your own words. 
    If the answer is not available, say 'Information not found'.
    Patient Records:
    {context_text}
    Question: {query}
    Answer:
    """
    answer = gemini_llm(prompt)
    # sources = ", ".join([doc["metadata"]["filename"] for doc in top_docs])
    # Render HTML with answer (only the LLM answer is shown in the chat-box)
    return templates.TemplateResponse("chat.html", {"request": request, "answer": answer, 
        "query_text": query})
