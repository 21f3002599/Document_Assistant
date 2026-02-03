import sys
import subprocess
import requests
import json
import numpy as np
from typing import List, Dict
from pathlib import Path
import PyPDF2
from sentence_transformers import SentenceTransformer, util
import warnings

warnings.filterwarnings('ignore')

# CONFIGURATION
OLLAMA_API_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "gemma3:4b"  

# DEPENDENCY CHECK 
def check_dependencies():
    required = {'PyPDF2': 'PyPDF2', 'sentence_transformers': 'sentence-transformers', 
                'numpy': 'numpy', 'requests': 'requests', 'torch': 'torch'}
    for module, pip_name in required.items():
        try:
            __import__(module)
        except ImportError:
            print(f"Installing {pip_name}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", pip_name])

check_dependencies()

class DocumentAssistant:
    def __init__(self):
        print("\nInitializing Embedding Model (MiniLM)...")
        self.embedder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        self.chunks: List[Dict] = [] 
        self.chunk_embeddings = None
        self.document_loaded = False
        self.filename = ""
        self.total_pages = 0
        
        self.chunk_size = 800      
        self.chunk_overlap = 100   
        self.similarity_threshold = 0.15 

    def extract_text_with_pages(self, file_path):
        path = Path(file_path)
        if not path.exists(): return None, "File not found."
        pages_data = []
        try:
            if path.suffix.lower() == '.pdf':
                with open(path, 'rb') as f:
                    reader = PyPDF2.PdfReader(f)
                    self.total_pages = len(reader.pages)
                    for i, page in enumerate(reader.pages):
                        text = page.extract_text()
                        if text:
                            clean_text = " ".join(text.split())
                            pages_data.append({'text': clean_text, 'page': i + 1})
            else:
                with open(path, 'r', encoding='utf-8') as f:
                    text = f.read()
                    pages_data.append({'text': " ".join(text.split()), 'page': 1})
                    self.total_pages = 1
            return pages_data, None
        except Exception as e: return None, str(e)

    def chunk_text(self, pages_data):
        all_chunks = []
        for entry in pages_data:
            text = entry['text']
            page_num = entry['page']
            start = 0
            while start < len(text):
                end = start + self.chunk_size
                if end < len(text):
                    period_pos = text.rfind('.', start, end)
                    if period_pos != -1 and period_pos > start + (self.chunk_size // 2):
                        end = period_pos + 1
                chunk_text = text[start:end].strip()
                if len(chunk_text) > 50:
                    all_chunks.append({'text': chunk_text, 'page': page_num})
                start = end - self.chunk_overlap
        return all_chunks

    def load_document(self, file_path):
        print(f"Loading {file_path}...")
        pages_data, error = self.extract_text_with_pages(file_path)
        if error:
            print(f"Error: {error}")
            return False
        if not pages_data:
            print("Error: Document seems empty.")
            return False
        self.filename = Path(file_path).name
        self.chunks = self.chunk_text(pages_data)
        
        text_only = [c['text'] for c in self.chunks]
        self.chunk_embeddings = self.embedder.encode(text_only, convert_to_tensor=True)
        self.document_loaded = True
        print(f"Document loaded ({self.total_pages} pages) and indexed!\n")
        return True

    def query_ollama(self, prompt):
        data = {
            "model": OLLAMA_MODEL, 
            "prompt": prompt, 
            "stream": False,
            "options": {
                "temperature": 0.0  
            }
        }
        try:
            response = requests.post(OLLAMA_API_URL, json=data)
            if response.status_code == 200: return response.json()['response']
            else: return f"Error from Ollama: {response.text}"
        except Exception as e: return f"Connection error: {e}"

    def ask(self, question):
        if not self.document_loaded: return "Please load a document first."

        # Page count question
        q_lower = question.lower()
        if "pages" in q_lower and ("total" in q_lower or "how many" in q_lower or "count" in q_lower or "number" in q_lower):
            return f"This document has a total of {self.total_pages} pages."

        # Search
        query_embedding = self.embedder.encode(question, convert_to_tensor=True)
        scores = util.cos_sim(query_embedding, self.chunk_embeddings)[0]
        
        results = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:3]
        best_score = results[0][1].item()
        
        # Hallucination Threshold
        if best_score < self.similarity_threshold:
            return "I don't know. (The information does not appear to be in this document)."

        # Format Context
        context_text = ""
        found_pages = set()
        for idx, score in results:
            chunk = self.chunks[idx]
            found_pages.add(str(chunk['page']))
            context_text += f"--- PAGE {chunk['page']} ---\n{chunk['text']}\n\n"

        prompt = f"""You are an intelligent research assistant.
        
DOCUMENT: {self.filename} (Total Pages: {self.total_pages})

CONTEXT:
{context_text}

USER QUESTION: {question}

INSTRUCTIONS:
1. Answer using ONLY the Context provided.
2. If the context does not contain the answer, say exactly "I don't know." and STOP. Do not explain.
3. Mention specific page numbers in your answer if relevant.

ANSWER:"""

        raw_answer = self.query_ollama(prompt)
        
        # Add Citations if valid answer
        refusal_phrases = ["i don't know", "no information", "cannot answer", "not found"]
        if not any(phrase in raw_answer.lower() for phrase in refusal_phrases):
            return raw_answer + f"\n\n(Source Pages: {', '.join(sorted(found_pages))})"
        return raw_answer

def main():
    print("="*60)
    print(f"   Document Assistant (Powered by {OLLAMA_MODEL})")
    print("="*60)
    
    try:
        requests.get("http://localhost:11434/")
    except:
        print("Error: Ollama is not running. Please start Ollama and try again.")
        return

    assistant = DocumentAssistant()
    
    while True:
        f = input("\nEnter PDF path (or 'q'): ").strip().strip('"')
        if f.lower() == 'q': return
        if assistant.load_document(f): break
            
    while True:
        q = input("\nQuestion: ").strip()
        if q.lower() == 'q': break
        if q: print(f"\n>> {assistant.ask(q)}")

if __name__ == "__main__":
    main()