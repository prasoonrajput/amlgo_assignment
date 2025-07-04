
import faiss
import numpy as np
import pickle
import os
from typing import List, Tuple, Dict, Any
from sentence_transformers import SentenceTransformer


class VectorDatabase:
    def __init__(self, embedding_model_name: str = "all-MiniLM-L6-v2"):
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.index = None
        self.chunks = []
        self.embeddings = None
        self.model_name = embedding_model_name
    
    def create_embeddings(self, chunks: List[Dict[str, Any]]) -> np.ndarray:
        """Generate embeddings for document chunks."""
        texts = [chunk['text'] for chunk in chunks]
        embeddings = self.embedding_model.encode(texts, show_progress_bar=True)
        return embeddings
    
    def build_index(self, chunks: List[Dict[str, Any]]):
        """Build FAISS index from chunks."""
        self.chunks = chunks
        self.embeddings = self.create_embeddings(chunks)
        
        # Create FAISS index
        dimension = self.embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(self.embeddings.astype('float32'))
        
        print(f"Built FAISS index with {len(chunks)} chunks")
        print(f"Embedding dimension: {dimension}")
    
    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar chunks."""
        if self.index is None:
            raise ValueError("Index not built. Call build_index() first.")
        
        # Generate query embedding
        query_embedding = self.embedding_model.encode([query])
        
        # Search
        distances, indices = self.index.search(query_embedding.astype('float32'), k)
        
        # Return results with similarity scores
        results = []
        for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
            if idx != -1:  # Valid result
                chunk = self.chunks[idx].copy()
                chunk['similarity_score'] = float(distance)
                chunk['rank'] = i + 1
                results.append(chunk)
        
        return results
    
    def save_index(self, index_path: str, metadata_path: str):
        """Save FAISS index and metadata."""
        os.makedirs(os.path.dirname(index_path), exist_ok=True)
        os.makedirs(os.path.dirname(metadata_path), exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.index, index_path)
        
        # Save metadata
        metadata = {
            'chunks': self.chunks,
            'embeddings': self.embeddings.tolist(),
            'model_name': self.model_name
        }
        
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)
    
    def load_index(self, index_path: str, metadata_path: str):
        """Load FAISS index and metadata."""
        # Load FAISS index
        self.index = faiss.read_index(index_path)
        
        # Load metadata
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)
        
        self.chunks = metadata['chunks']
        self.embeddings = np.array(metadata['embeddings'])
        self.model_name = metadata['model_name']
        
        print(f"Loaded FAISS index with {len(self.chunks)} chunks")

