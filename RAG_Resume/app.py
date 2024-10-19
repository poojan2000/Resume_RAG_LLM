from flask import Flask, render_template, request, jsonify
from huggingface_hub import InferenceClient
import torch
import re
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer, util

app = Flask(__name__)

# Initialize the client for Hugging Face's model
hf_token = "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"
client = InferenceClient(api_key=hf_token)

# Load the sentence transformer model for similarity search
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# Load and preprocess the resume
pdf_path = "Pooja Niranjan_Resume_2024.pdf"
resume_text = ""
resume_chunks = []
resume_embeddings = []

def get_pred(question):
    messages = [
        {"role": "user", "content": question}
    ]
    stream = client.chat.completions.create(
        model="meta-llama/Meta-Llama-3-8B-Instruct",
        messages=messages,
        temperature=0.5,
        max_tokens=1024,
        top_p=0.7
    )
    return stream.choices[0].message.content

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page_num in range(doc.page_count):
        page = doc.load_page(page_num)
        text += page.get_text()
    return text

def split_text_into_chunks(text, chunk_size=100):
    words = re.findall(r'\w+', text)
    chunks = [' '.join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]
    return chunks

def retrieve_top_chunks(question, resume_embeddings, resume_chunks, top_n=5):
    question_embedding = embedder.encode(question, convert_to_tensor=True)
    similarities = util.pytorch_cos_sim(question_embedding, resume_embeddings)
    top_n_indices = torch.topk(similarities, top_n).indices[0].tolist()
    top_chunks = [resume_chunks[i] for i in top_n_indices]
    combined_context = " ".join(top_chunks)
    return combined_context

def generate_answer(question, context):
    prompt = f"Context: {context}\nQuestion: {question}\nAnswer:"
    answer = get_pred(prompt)
    return answer

def answer_question(question):
    combined_context = retrieve_top_chunks(question, resume_embeddings, resume_chunks, top_n=5)
    answer = generate_answer(question, combined_context)
    return answer

# Preprocess the resume text and embeddings once at startup
resume_text = extract_text_from_pdf(pdf_path)
resume_chunks = split_text_into_chunks(resume_text)
resume_embeddings = embedder.encode(resume_chunks, convert_to_tensor=True)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask_question():
    data = request.get_json()
    question = data.get("question")
    answer = answer_question(question)
    return jsonify({"answer": answer})

if __name__ == "__main__":
    app.run(debug=True)
