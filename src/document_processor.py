
import re
import json
import os
from typing import List, Dict, Any
from langchain.text_splitter import RecursiveCharacterTextSplitter
import numpy as np

class DocumentProcessor:
    def __init__(self, chunk_size: int = 200, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
        )
    
    def clean_text(self, text: str) -> str:
        """Clean and preprocess text."""
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep punctuation
        text = re.sub(r'[^\w\s\.,!?;:()\-"]', '', text)
        
        # Remove multiple consecutive punctuation
        text = re.sub(r'([.!?]){2,}', r'\1', text)
        
        return text.strip()
    
    def chunk_document(self, text: str) -> List[Dict[str, Any]]:
        """Split document into chunks with metadata."""
        cleaned_text = self.clean_text(text)
        chunks = self.text_splitter.split_text(cleaned_text)
        
        chunked_docs = []
        for i, chunk in enumerate(chunks):
            chunked_docs.append({
                'id': f'chunk_{i}',
                'text': chunk,
                'word_count': len(chunk.split()),
                'char_count': len(chunk),
                'chunk_index': i
            })
        
        return chunked_docs
    
    def save_chunks(self, chunks: List[Dict[str, Any]], filepath: str):
        """Save chunks to JSON file."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(chunks, f, indent=2, ensure_ascii=False)
    
    def load_chunks(self, filepath: str) -> List[Dict[str, Any]]:
        """Load chunks from JSON file."""
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)