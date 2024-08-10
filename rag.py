import os
import json
import re
import nltk
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import google.generativeai as genai
import chromadb
from typing import List, Tuple

# Download NLTK data
nltk.download('punkt', quiet=True)
from nltk.tokenize import sent_tokenize
print("NLTK data downloaded successfully.")
# Set Tesseract path for Windows
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

class GeminiEmbeddingFunction:
    """
    Custom embedding function using the Gemini AI API for document retrieval.
    """
    def __init__(self):
        gemini_api_key = os.getenv("GEMINI_API_KEY")
        if not gemini_api_key:
            raise ValueError("Gemini API Key not provided. Please set GEMINI_API_KEY as an environment variable")
        genai.configure(api_key=gemini_api_key)
        self.model = "models/text-embedding-004"
        self.title = "Custom query"

    def __call__(self, input: List[str]) -> List[List[float]]:
        embeddings = []
        for content in input:
            if content.strip():
                sub_chunks = self.split_into_sub_chunks(content, max_chars=1000)
                for sub_chunk in sub_chunks:
                    embedding = genai.embed_content(model=self.model, content=sub_chunk, task_type="retrieval_document", title=self.title)["embedding"]
                    embeddings.append(embedding)
        return embeddings

    @staticmethod
    def split_into_sub_chunks(text: str, max_chars: int) -> List[str]:
        """Split text into sub-chunks within the allowed size limit."""
        sub_chunks = []
        current_chunk = []
        current_length = 0

        for word in text.split():
            word_length = len(word) + 1
            if current_length + word_length > max_chars:
                sub_chunks.append(" ".join(current_chunk))
                current_chunk = [word]
                current_length = word_length
            else:
                current_chunk.append(word)
                current_length += word_length

        if current_chunk:
            sub_chunks.append(" ".join(current_chunk))

        return sub_chunks

def pdf_to_images(pdf_path: str) -> List[Image.Image]:
    """Convert PDF pages to images."""
    doc = fitz.open(pdf_path)
    return [Image.frombytes("RGB", [page.get_pixmap().width, page.get_pixmap().height], page.get_pixmap().samples)
            for page in doc]

def extract_text_from_images(images: List[Image.Image]) -> str:
    """Extract text from a list of images using OCR."""
    return " ".join(pytesseract.image_to_string(img) for img in images)

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text from a PDF file."""
    return extract_text_from_images(pdf_to_images(pdf_path))

def preprocess_text(text: str) -> str:
    """Preprocess the extracted text."""
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s]', '', text)
    return text.lower()

def create_chunks(text: str, max_chunk_size: int = 512) -> List[str]:
    """Create chunks from the preprocessed text."""
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = []
    current_length = 0
    
    for sentence in sentences:
        sentence_length = len(sentence.split())
        if current_length + sentence_length <= max_chunk_size:
            current_chunk.append(sentence)
            current_length += sentence_length
        else:
            chunks.append(' '.join(current_chunk))
            current_chunk = [sentence]
            current_length = sentence_length
    
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks

def create_chroma_db(documents: List[str], embeddings: List[List[float]], path: str, name: str) -> Tuple[chromadb.Collection, str]:
    """Creates a Chroma database using the provided documents and embeddings."""
    client = chromadb.Client()
    db = client.create_collection(name=name)
    
    for i, (document, embedding) in enumerate(zip(documents, embeddings)):
        db.add(documents=document, ids=str(i), embeddings=[embedding])

    return db, name

def load_chroma_db(name: str) -> chromadb.Collection:
    """Loads a ChromaDB collection by name."""
    client = chromadb.Client()
    return client.get_collection(name=name)

def get_relevant_passage(query: str, db: chromadb.Collection) -> str:
    """Retrieves the most relevant passage from the ChromaDB based on the user query."""
    embedding_function = GeminiEmbeddingFunction()
    query_embedding = embedding_function([query])[0]
    results = db.query(query_embeddings=[query_embedding], n_results=1)
    return results['documents'][0][0]  # Return the single most relevant passage

def process_pdf_and_create_db(pdf_path: str, db_path: str, db_name: str) -> chromadb.Collection:
    """Process PDF, create chunks and embeddings, and initialize the database."""
    # Extract and preprocess text from PDF
    text = extract_text_from_pdf(pdf_path)
    preprocessed_text = preprocess_text(text)
    chunks = create_chunks(preprocessed_text)

    # Create embeddings
    embedding_function = GeminiEmbeddingFunction()
    embeddings = embedding_function(chunks)

    # Create and return ChromaDB
    db, _ = create_chroma_db(documents=chunks, embeddings=embeddings, path=db_path, name=db_name)
    return db

def generate_single_line_answer(query: str, context: str) -> str:
    """Generate a single-line answer using Gemini AI."""
    model = genai.GenerativeModel('gemini-pro')
    prompt = f"Based on the following context, provide a single-line answer to the query. Context: {context}\nQuery: {query}\nSingle-line answer:"
    response = model.generate_content(prompt)
    return response.text.strip()

def main():
    # Define paths and names
    pdf_path = r"D:\rag-demo\VR20-_IT_Syllabus.pdf"
    db_path = "C:/Repos/RAG/contents"
    db_name = "rag_experiment"

    # Process PDF and create database (or load if it exists)
    try:
        db = load_chroma_db(db_name)
        print(f"Loaded existing ChromaDB Collection '{db_name}'.")
    except ValueError:
        db = process_pdf_and_create_db(pdf_path, db_path, db_name)
        print(f"Created new ChromaDB Collection '{db_name}'.")

    # Example usage of retrieval and single-line answer generation
    query = input("Enter ypur regular expression query?")
    relevant_passage = get_relevant_passage(query, db)
    answer = generate_single_line_answer(query, relevant_passage)

    print("\nQuery:", query)
    print("Answer:", answer)

if __name__ == "__main__":
    main()