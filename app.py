from flask import Flask, request, render_template, redirect, url_for
import os
import uuid
from werkzeug.utils import secure_filename

from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import LLMChain, RetrievalQA
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader


# App setup
app = Flask(__name__)


# creating 'uploads' folder which will contain resume pdfs 
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
# it is a dictionary like---
'''app.config = {
    "UPLOAD_FOLDER": "uploads"
}'''



# Text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1200,
    chunk_overlap=100,
)



# Embeddings 
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-MiniLM-L3-v2"
)


# LLM (Groq – free)
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0,
    api_key=os.getenv("GROQ_API_KEY")
)



# Prompt + Chain (Resume Summary)
resume_summary_template = """
Role: You are an AI Career Coach.

Task: Given the candidate's resume, provide a structured summary including:

- Career Objective
- Skills and Expertise
- Professional Experience
- Educational Background
- Notable Achievements

Guidelines:
Be concise, professional, and highlight strengths clearly.

Most important note:
If user asks the question out of the context of his uploaded resume, then say"I can't provide you the information based on your provided resume"

Resume:
{resume}
"""

resume_prompt = PromptTemplate(
    input_variables=["resume"],
    template=resume_summary_template,
)

resume_analysis_chain = LLMChain(
    llm=llm,
    prompt=resume_prompt,
)



# In-memory vector store
vectorstores = {} # resume dictionary
'''vectorstores = {
    "resume_id_1": FAISS_object_for_resume_1,
    "resume_id_2": FAISS_object_for_resume_2,
    ...
}'''


# RAG Q&A function
def perform_qa(query, resume_id):

    db = vectorstores.get(resume_id)
    # if vectorstore does not have this resume_id
    if not db:
        return "Session expired. Please upload your resume again."

    retriever = db.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=False
    )

    result = qa_chain.invoke({"query": query})
    return result["result"]


# -----------------------------
# Routes
# -----------------------------
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload_file():

    if "file" not in request.files:
        return redirect(url_for("index"))

    file = request.files["file"]

    if file.filename == "":
        return redirect(url_for("index"))

    resume_id = str(uuid.uuid4())
    filename = secure_filename(file.filename)
    unique_filename = f"{resume_id}__{filename}"
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], unique_filename)
    file.save(file_path)

    # Load PDF
    loader = PyPDFLoader(file_path)
    docs = loader.load()

    # it occurs only if the uploaded pdf has any picture(picture cannot be embedded and stripped)
    if not docs or not any(doc.page_content.strip() for doc in docs):
        return "This resume appears to be scanned or empty. Please upload a text-based PDF.", 400

    # Split text in chunks
    chunks = text_splitter.split_documents(docs)
    if not chunks:
        return "Could not split PDF into chunks.", 400

    # Create vector store (in-memory)
    vectorstores[resume_id] = FAISS.from_documents(chunks, embeddings)


    # Full resume text 
    resume_text = "\n".join(doc.page_content for doc in docs)

    resume_analysis = resume_analysis_chain.invoke({"resume": resume_text})["text"]

    return render_template(
        "results.html",
        resume_analysis=resume_analysis,
        resume_id=resume_id,
        filename=filename
    )


@app.route("/ask", methods=["GET", "POST"])
def ask_query():

    resume_id = request.values.get("resume_id")

    if request.method == "POST":
        query = request.form["query"]
        answer = perform_qa(query, resume_id)

        return render_template(
            "qa_results.html",
            query=query,
            result=answer,
            resume_id=resume_id
        )

    return render_template("ask.html", resume_id=resume_id)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

