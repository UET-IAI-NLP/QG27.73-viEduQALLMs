import fitz  # PyMuPDF
import os
from datasets import Dataset
from tqdm import tqdm
def extract_text_from_pdf(pdf_path):
    """Extracts text from a PDF file using PyMuPDF."""
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()
        return text
    except Exception as e:
        print(f"Error processing {pdf_path}: {e}")
        return None

# Example usage (assuming you have PDFs in the 'nhasachmienphi_pdfs' folder)
pdf_folder = "/workspace/thviet/LLMs/Monolingual/QG27.73-viEduQALLMs/nhasachmienphi_pdfs"  # Update this path
chunks = []
i = 0
for filename in tqdm(os.listdir(pdf_folder)):
    if filename.endswith(".pdf"):
        pdf_path = os.path.join(pdf_folder, filename)
        extracted_text = extract_text_from_pdf(pdf_path)

        if extracted_text:
            words = extracted_text.split()  # Split by whitespace into words
            total_words = len(words)
            avg_words = total_words / float(100)
            last = 0.0

            while last < total_words:
                chunk = words[int(last):int(last + avg_words)]
                chunks.append({'text':' '.join(chunk)})
                last += avg_words
        #print(len(chunks))
dataset = Dataset.from_dict({'text': [chunk['text'] for chunk in chunks]})
print(dataset)
dataset.push_to_hub('zerostratos/books', token = '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')
