#  AI Career Coach (Groq-powered)

An AI-powered web application that analyzes resumes and acts as a **career coach**. Users can upload a resume (PDF), get an AI-generated summary, and ask contextual questions about their profile using **Retrieval-Augmented Generation (RAG)**.

This project is built as a **portfolio-grade full-stack AI application**, deployed for free on **Render**, and follows industry best practices (environment variables, no hardcoded secrets, stateless backend).

---

##  Features

-  Upload a resume (text-based PDF)
-  AI-generated resume summary Ask follow-up questions about the resume
-  Semantic search using vector embeddings
-  Retrieval-Augmented Generation (RAG)
-  Secure API key handling using environment variables

---

##  Tech Stack
### Backend

- **Python (Flask)** – Web framework
- **LangChain** – LLM orchestration
- **Groq API** – Fast, free LLM inference
- **FAISS** – Vector database for similarity search
- **Sentence-Transformers** – HuggingFace embeddings

### Models

- **LLM:** `llama-3.1-8b-instant` (Groq)
- **Embeddings:** `sentence-transformers/all-MiniLM-L6-v2`

### Deployment

- **Render** (Free Web Service)
- **Gunicorn** – Production WSGI server

---

##  How It Works (Architecture)

1. User uploads a resume (PDF)
2. Resume text is extracted and split into chunks
3. Chunks are converted into embeddings using HuggingFace
4. Embeddings are stored in a FAISS vector index
5.   - Resume summary → generated using Groq LLM
     - User questions → answered using RAG (FAISS + Groq)

---

##  Limitations (Free Tier)

- App sleeps after inactivity
- Vector indexes reset on restart
- No persistent storage

---

##  Why This Project Matters

This project demonstrates:

- Real-world LLM integration
- RAG architecture
- Secure secret management
- Deployment-ready Flask app
- Understanding of production constraints

---

##  Author

**Sangam Sharma**\


