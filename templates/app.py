# app.py

import os
import numpy as np
from flask import Flask, render_template, request, jsonify
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import PyPDF2
import re
import logging
import google.generativeai as genai
import dotenv

dotenv.load_dotenv()

# Config
GEMINI_API_KEY = os.environ.get("AIzaSyBwmkSiS-BE1I_2iEwGYBFBump_07mplyU")
GEMINI_MODEL = "gemini-2.0-flash"
PDF_PATH = "static/docs/manual.pdf"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
SECRET_KEY = os.environ.get("SECRET_KEY", "dev")

logging.basicConfig(level=logging.INFO)
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel(GEMINI_MODEL)

app = Flask(__name__)
app.secret_key = SECRET_KEY

class PDFChatbot:
    def __init__(self, pdf_path):
        self.pdf_path = pdf_path
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL)
        self.chunks = []
        self.embeddings = None
        self.load_and_process_pdf()

    def extract_text(self):
        text = ""
        with open(self.pdf_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            for i, page in enumerate(reader.pages):
                try:
                    content = page.extract_text()
                    if content:
                        text += f"\n--- Page {i+1} ---\n{content}"
                except Exception as e:
                    logging.warning(f"Failed to extract text from page {i}: {e}")
        return text

    def clean_text(self, text):
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"--- Page \d+ ---", "", text)
        return text.strip()

    def chunk_text(self, text, chunk_size=800):
        sentences = re.split(r'[.!?]+', self.clean_text(text))
        chunks, current = [], ""
        for sentence in sentences:
            if len(current) + len(sentence) > chunk_size:
                chunks.append(current.strip())
                current = sentence
            else:
                current += " " + sentence
        if current:
            chunks.append(current.strip())
        return [chunk for chunk in chunks if len(chunk) > 50]

    def load_and_process_pdf(self):
        logging.info("Processing predefined PDF...")
        text = self.extract_text()
        if not text:
            raise ValueError("No text extracted from PDF")
        self.chunks = self.chunk_text(text)
        self.embeddings = self.embedding_model.encode(self.chunks)
        logging.info(f"Loaded {len(self.chunks)} chunks.")

    def find_relevant_chunks(self, query, top_k=3):
        q_embedding = self.embedding_model.encode([query])
        sims = cosine_similarity(q_embedding, self.embeddings)[0]
        indices = np.argsort(sims)[-top_k:][::-1]
        return [(self.chunks[i], sims[i]) for i in indices]

    def answer_question(self, question):
        chunks = self.find_relevant_chunks(question)
        if not chunks:
            return {"answer": "No relevant content found.", "sources": [], "confidence": 0}
        context = "\n\n".join([c for c, _ in chunks])
        prompt = f"""Use the following context to answer the question:

Context:
{context}

Question: {question}

Answer:"""
        try:
            response = gemini_model.generate_content(prompt)
            return {
                "answer": response.text.strip(),
                "sources": [c[:200] + "..." for c, _ in chunks],
                "confidence": float(np.mean([s for _, s in chunks]))
            }
        except Exception as e:
            logging.error(f"Gemini error: {e}")
            return {"answer": "Error contacting LLM.", "sources": [], "confidence": 0}

# Single instance chatbot
chatbot = PDFChatbot(PDF_PATH)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/ask", methods=["POST"])
def ask():
    data = request.get_json()
    question = data.get("question", "").strip()
    if not question:
        return jsonify({"error": "No question provided."}), 400
    result = chatbot.answer_question(question)
    return jsonify(result)

@app.route("/api/status")
def status():
    return jsonify({
        "status": "ready",
        "chunks": len(chatbot.chunks)
    })

if __name__ == "__main__":
    app.run(debug=True)
