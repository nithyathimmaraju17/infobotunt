import os
from flask import Flask, render_template, request
import PyPDF2
from sentence_transformers import SentenceTransformer
import faiss
from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer

app = Flask(__name__)

# Load Hugging Face QA model
model_name = "google/flan-t5-large" 
tokenizer = AutoTokenizer.from_pretrained(model_name)
qa_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
qa_pipeline = pipeline("text2text-generation", model=qa_model, tokenizer=tokenizer)

# Extract text from PDF
def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
    return text

# Preprocess and split text into chunks
def preprocess_text(text, chunk_size=512):
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

# Load retriever model
retriever_model = SentenceTransformer('all-MiniLM-L6-v2')

# Convert chunks to embeddings and store in FAISS index
def create_faiss_index(chunks):
    embeddings = retriever_model.encode(chunks, convert_to_tensor=True)
    faiss_index = faiss.IndexFlatL2(embeddings.shape[1])  # Changed `index` to `faiss_index`
    faiss_index.add(embeddings.cpu().numpy())
    return faiss_index, chunks

# Query the FAISS index
def retrieve_relevant_chunks(query, faiss_index, chunks, top_k=5):  # Changed `index` to `faiss_index`
    query_embedding = retriever_model.encode([query], convert_to_tensor=True)
    _, indices = faiss_index.search(query_embedding.cpu().numpy(), top_k)  # Changed `index` to `faiss_index`
    return [chunks[i] for i in indices[0]]

# Generate response using retrieved chunks via Hugging Face model
def generate_response(query, retrieved_chunks):
    context = " ".join(retrieved_chunks)
    input_text = f"Context: {context}\n\nQuestion: {query}\n\nAnswer:"
    response = qa_pipeline(input_text, max_length=512)[0]["generated_text"]
    return response

# Initialize data
pdf_path = 'ISSS information.pdf'  # Make sure this file is in the same directory
text = extract_text_from_pdf(pdf_path)
chunks = preprocess_text(text)
faiss_index, chunks = create_faiss_index(chunks)  # Changed `index` to `faiss_index`

# Flask routes
@app.route("/", methods=["GET", "POST"])
def index():
    answer = ""
    if request.method == "POST":
        query = request.form["query"]
        retrieved_chunks = retrieve_relevant_chunks(query, faiss_index, chunks)  # Changed `index` to `faiss_index`
        answer = generate_response(query, retrieved_chunks)
    return render_template("index.html", answer=answer)

if __name__ == "__main__":
    app.run(debug=True)
