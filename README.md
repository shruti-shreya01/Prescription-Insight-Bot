# 🩺 Prescription-Insight-Bot

**From handwritten prescription → Extracted insights → Summarized medical data → Interactive RAG chatbot**

A professional-grade AI pipeline to extract, summarize, and interact with handwritten medical prescriptions, leveraging OCR, semantic search, LLM summarization, and RAG-based conversational AI.


![Python](https://img.shields.io/badge/Python-%2314354C.svg?style=for-the-badge&logo=python&logoColor=white)  ![FastAPI](https://img.shields.io/badge/FastAPI-%2300B3C4.svg?style=for-the-badge&logo=fastapi&logoColor=white)  ![Flask](https://img.shields.io/badge/Flask-%23000000.svg?style=for-the-badge&logo=flask&logoColor=white)  ![HTML5](https://img.shields.io/badge/HTML5-%23E34F26.svg?style=for-the-badge&logo=html5&logoColor=white)  ![CSS3](https://img.shields.io/badge/CSS3-%231572B6.svg?style=for-the-badge&logo=css3&logoColor=white)  ![ChromaDB](https://img.shields.io/badge/ChromaDB-%23FF5733.svg?style=for-the-badge&logoColor=white)  ![LangChain](https://img.shields.io/badge/LangChain-%2300FFAA.svg?style=for-the-badge&logoColor=white)  ![RAG](https://img.shields.io/badge/RAG-%23FF6600.svg?style=for-the-badge&logoColor=white)  ![Sentence-Transformers](https://img.shields.io/badge/Sentence--Transformers-%230000FF.svg?style=for-the-badge&logoColor=white)  ![NumPy](https://img.shields.io/badge/NumPy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)  ![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)  ![Google Gemini](https://img.shields.io/badge/Google_Gemini-%23F4B400.svg?style=for-the-badge&logo=google&logoColor=white)  ![python-dotenv](https://img.shields.io/badge/python--dotenv-%23000000.svg?style=for-the-badge&logoColor=white)  ![Docker](https://img.shields.io/badge/Docker-%230db7ed.svg?style=for-the-badge&logo=docker&logoColor=white)  

---

## 🚀 Features

- **Handwritten Prescription Extraction (Task 1)**  
  Uses `naver-clova-ix/donut-base-finetuned-cord-v2` to extract structured text from handwritten prescriptions.  

- **Medical Records Summarization (Task 2)**  
  Summarizes extracted notes with relevant fields (`Patient`, `Diagnosis`, `Treatment`, `Followup`) using **Google Gemini 2.0 Flash** LLM.  

- **Hybrid Semantic Search (ChromaDB + Keyword Search)**  
  - Keyword-based exact match boost.  
  - Semantic similarity search using embeddings from `all-MiniLM-L6-v2`.  
  - Ranks results prioritizing keywords, then semantic similarity.

- **RAG Chatbot (Task 3)**  
  FastAPI-based interactive medical assistant:
  - Retrieves relevant context using hybrid search.  
  - Generates concise answers using Gemini LLM.  
  - Only returns validated responses; no hallucinations.

- **Dockerized Deployment (Task 4)**  
  Containerized with Python 3.13-slim, ready for production deployment.  

- **Secure Environment**  
  Uses `.env` file to manage API keys safely.

---

## 📦 Project Structure

```bash
assessment-infraintelai/
│
├─ task1/                     # Handwritten text extraction
│   ├─ mycleaned_texts/       # Cleaned OCR text files
│   ├─ doc_text_extract.ipynb # Notebook for text extraction
│   └─ fastapi_chromadb.py    # Keyword search API
│
├─ task2/                     # Summarization pipeline
│   ├─ llm_summarize.ipynb    # Notebook for LLM summarization
│   ├─ fastapi_summarize.py   # Optional API interface for summarization
│   └─ summaries.json         # Output summaries
│
├─ task3/                     # RAG chatbot
│   ├─ rag.py                 # FastAPI app
│   ├─ rag.ipynb              # Notebook for prototyping
│   ├─ chroma_rag_db/         # Chroma vector store
│   ├─ templates/
│   │   └─ chat.html
│   └─ static/
│       └─ style.css
│
├─ task4/ (Docker setup)
│   └─ dockerfile
│
├─ requirements.txt
├─ .gitignore
└─ README.md
```

## Architecture
```bash
flowchart TD
    A[📝 Handwritten Prescription] --> B[🔍 OCR Text Extraction with DONUT]
    B --> C[🧠 Summarization with Gemini LLM]
    C --> D[📂 ChromaDB + Embeddings (all-MiniLM-L6-v2)]
    D --> E[🤖 RAG Chatbot API]
    E --> F[💻 FastAPI Frontend UI]
```

## 🚀 Tech Stack

- **OCR/Text Extraction**: `naver-clova-ix/donut-base-finetuned-cord-v2`  
- **Summarization LLM**: Google Gemini 2.0 Flash API  
- **Vector Search**: ChromaDB, `sentence-transformers/all-MiniLM-L6-v2`  
- **Web Framework**: FastAPI + Jinja2 templates  
- **Deployment**: Docker, `.env` for API keys  
- **Python Packages**: `sentence-transformers`, `scikit-learn`, `numpy`, `python-dotenv`, `google-generativeai`  

---

## 📝 Task 1: Handwritten Prescription Extraction

- Uses the DONUT model for text extraction.  
- Converts images to structured text.  
- Output stored in `task1/mycleaned_texts/` as `.txt` files.

```python
from donut import DonutModel

# Load pre-trained DONUT model
model = DonutModel.from_pretrained("naver-clova-ix/donut-base-finetuned-cord-v2")

# Extract text from an image
output = model.extract_text(image)
```
## 🚀 Tech Stack

- **OCR/Text Extraction**: `naver-clova-ix/donut-base-finetuned-cord-v2`  
- **Summarization LLM**: Google Gemini 2.0 Flash API  
- **Vector Search**: ChromaDB, `sentence-transformers/all-MiniLM-L6-v2`  
- **Web Framework**: FastAPI + Jinja2 templates  
- **Deployment**: Docker, `.env` for API keys  
- **Python Packages**: `sentence-transformers`, `scikit-learn`, `numpy`, `python-dotenv`, `google-generativeai`  

---

## 📝 Task 2: Summarization with Gemini LLM

- Loads `.txt` files from `task1/mycleaned_texts/`.
- Generates JSON summaries with keys:  `Patient`, `Diagnosis`, `Treatment`, `Followup`.
- Corrects common typos in clinical notes automatically.

```python
import os
import google.generativeai as genai

# Configure Gemini API key from environment
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
gemini_model = genai.GenerativeModel("gemini-2.0-flash")

# Prompt template
def make_prompt(note_text):
    return f"Extract fields from clinical note:\n{note_text}\nReturn JSON."
```
## 📝 Task 3: Mini RAG chatbot

- Embeddings computed using `all-MiniLM-L6-v2`.
- Hybrid search:
  -Keyword match boost.
  -Semantic similarity using cosine similarity.

```python
def hybrid_search(query, top_k=3):
    # Keyword match
    # Semantic similarity
    # Rank: keyword first, then semantic
    return results
```

- FastAPI endpoints:
  - `GET /` → Renders chat UI `(chat.html)`
  - `POST /chat` → Receives query, retrieves top docs, calls Gemini LLM, returns answer.

## Task 4: Docker Deployment

```python
FROM python:3.13-slim

WORKDIR /rag

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project
COPY . .

# Expose FastAPI port
EXPOSE 8000

# Start the FastAPI app
CMD ["uvicorn", "rag:app", "--host", "0.0.0.0", "--port", "8000"]
```
**Build and Run**
```python
docker build -t prescription-insight-bot .
docker run -d -p 8000:8000 --env-file .env prescription-insight-bot
```
## Usage

- Upload handwritten prescription images → Task1 OCR extracts text.

- Run Task2 summarizer → Generates JSON summaries.

- Query Task3 chatbot → Receives answers with context from summaries.
- **Sample queries**:
```bash
"What is the dosage for Tab. Metformin?"

"Summarize patient treatment instructions."

"List all follow-up instructions."
```

## 🧑‍💻 Author

**Shruti Shreya**  

This project is currently a **Proof of Concept (PoC)** for handwritten prescription analysis and RAG-based chatbot.  

- 🔗 [LinkedIn](https://www.linkedin.com/in/shruti-shreya-893789265/)
- 💻 [GitHub](https://github.com/shruti-shreya01)  
