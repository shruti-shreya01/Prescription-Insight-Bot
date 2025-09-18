import os
from fastapi import FastAPI, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
import chromadb
from sentence_transformers import SentenceTransformer

# ---------- Setup ----------
DATA_DIR = r"C:\Users\shruti shreya\Downloads\assessment-infraintelai\task1\mycleaned_texts"
COLLECTION_NAME = "prescriptions"

chroma_client = chromadb.PersistentClient(path="chroma_db")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Create or get collection
collection = chroma_client.get_or_create_collection(COLLECTION_NAME)

# Load and index documents (only once)
if len(collection.get()["documents"]) == 0:  
    docs, ids = [], []
    for idx, file in enumerate(os.listdir(DATA_DIR)):
        if file.endswith(".txt"):
            with open(os.path.join(DATA_DIR, file), "r", encoding="utf-8") as f:
                text = f.read()
                docs.append(text)
                ids.append(f"doc_{idx}")
    embeddings = embedding_model.encode(docs).tolist()
    collection.add(documents=docs, embeddings=embeddings, ids=ids)
    print("✅ Documents indexed into ChromaDB")

# ---------- FastAPI ----------
app = FastAPI(title="Medical Notes Search API")
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "results": []})

@app.get("/search", response_class=HTMLResponse)
def search_notes(request: Request, query: str = Query(..., description="Keyword or phrase to search")):
    results = []

    # 1. Keyword search
    keyword_matches = []
    all_docs = collection.get()["documents"]
    all_ids = collection.get()["ids"]

    for doc_id, doc in zip(all_ids, all_docs):
        if query.lower() in doc.lower():
            keyword_matches.append({"id": doc_id, "text": doc})

    if keyword_matches:
        results.extend(keyword_matches)

    # 2. Semantic search (only if needed)
    query_emb = embedding_model.encode([query]).tolist()
    semantic_results = collection.query(query_embeddings=query_emb, n_results=1)

    for doc_id, doc in zip(semantic_results["ids"][0], semantic_results["documents"][0]):
        if {"id": doc_id, "text": doc} not in results:
            results.append({"id": doc_id, "text": doc})

    return templates.TemplateResponse("index.html", {"request": request, "results": results, "query": query})
