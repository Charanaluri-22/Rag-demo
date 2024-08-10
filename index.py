# import fitz  # PyMuPDF
# import pytesseract
# from PIL import Image

# # If you're on Windows, set the Tesseract-OCR executable path
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# def pdf_to_images(pdf_path):
#     # Open the PDF
#     doc = fitz.open(pdf_path)
#     images = []
    
#     for page_num in range(len(doc)):
#         page = doc.load_page(page_num)
#         pix = page.get_pixmap()
#         img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
#         images.append(img)
    
#     return images

# def extract_text_from_images(images):
#     text = ""
    
#     for img in images:
#         text += pytesseract.image_to_string(img)
    
#     return text

# def extract_text_from_pdf(pdf_path):
#     images = pdf_to_images(pdf_path)
#     text = extract_text_from_images(images)
#     return text

# # Example usage
# pdf_path = r"D:\rag-demo\VR20-_IT_Syllabus.pdf"
# text = extract_text_from_pdf(pdf_path)
# print("Extracted Text:", text)

# # Save the extracted text to a file
# with open("output.txt", "w", encoding="utf-8") as file:
#     file.write(text)
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import re
import nltk

nltk.download('punkt')
from nltk.tokenize import sent_tokenize

# If you're on Windows, set the Tesseract-OCR executable path
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def pdf_to_images(pdf_path):
    # Open the PDF
    doc = fitz.open(pdf_path)
    images = []
    
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        pix = page.get_pixmap()
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        images.append(img)
    
    return images

def extract_text_from_images(images):
    text = ""
    
    for img in images:
        text += pytesseract.image_to_string(img)
    
    return text

def extract_text_from_pdf(pdf_path):
    images = pdf_to_images(pdf_path)
    text = extract_text_from_images(images)
    return text

def preprocess_text(text):
    # Remove special characters and multiple spaces
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s]', '', text)
    
    # Convert text to lowercase
    text = text.lower()
    
    return text

def create_chunks(text, max_chunk_size=512):
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

# Example usage
pdf_path = r"D:\rag-demo\VR20-_IT_Syllabus.pdf"
text = extract_text_from_pdf(pdf_path)
preprocessed_text = preprocess_text(text)
chunks = create_chunks(preprocessed_text)

# Save the chunks to a file
with open("chunks.txt", "w", encoding="utf-8") as file:
    for chunk in chunks:
        file.write(chunk + "\n\n")

print("Preprocessing and chunking completed. Check the 'chunks.txt' file for the results.")
