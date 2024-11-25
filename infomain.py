import os
import google.generativeai as genai
import PyPDF2
from sentence_transformers import SentenceTransformer
import faiss
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# Retrieve the API key from environment variables
api_key = os.getenv("GOOGLE_GENAI_API_KEY")

# Configure API Key directly
genai.configure(api_key=api_key)

# Create the model configuration
generation_config = {
  "temperature": 1,
  "top_p": 0.95,
  "top_k": 64,
  "max_output_tokens": 8192,
  "response_mime_type": "text/plain",
}

model = genai.GenerativeModel(
  model_name="gemini-1.5-flash",
  generation_config=generation_config,
  system_instruction="You are a helpful assistant designed to provide accurate and detailed information about international student visas, work authorizations, and related queries of International Student Services at the University of North Texas. Use the provided context to generate relevant and helpful responses to user queries.",
)

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

# Load models
retriever_model = SentenceTransformer('all-MiniLM-L6-v2')

# Convert chunks to embeddings and store in FAISS index
def create_faiss_index(chunks):
    embeddings = retriever_model.encode(chunks, convert_to_tensor=True)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings.cpu().numpy())
    return index, chunks

# Query the FAISS index
def retrieve_relevant_chunks(query, index, chunks, top_k=5):
    query_embedding = retriever_model.encode([query], convert_to_tensor=True)
    _, indices = index.search(query_embedding.cpu().numpy(), top_k)
    return [chunks[i] for i in indices[0]]

# Generate response using retrieved chunks via Google Gemini
def generate_response(query, retrieved_chunks):
    context = " ".join(retrieved_chunks)
    input_text = f"Context: {context}\n\nQuestion: {query}\n\nAnswer:"

    chat_session = model.start_chat(
        history=[
            {
                "role": "user",
                "parts": [input_text],
            },
        ]
    )

    response = chat_session.send_message(query)
    return response.text

# Main workflow
pdf_path = 'ISSS information.pdf'
text = extract_text_from_pdf(pdf_path)
chunks = preprocess_text(text)
index, chunks = create_faiss_index(chunks)

st.title("International Student Information")

query = st.text_input("Enter your question:")
if query:
    retrieved_chunks = retrieve_relevant_chunks(query, index, chunks)
    response = generate_response(query, retrieved_chunks)
    st.write(response)
