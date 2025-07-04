
from typing import List, Dict, Any, Generator
from .document_processor import DocumentProcessor
from .vector_database import VectorDatabase
from .llm_handler import LLMHandler
import pdfplumber

class RAGPipeline:
    def __init__(self, llm_model: str,
                 embedding_model: str = "all-MiniLM-L6-v2",
                 chunk_size: int = 200,
                 chunk_overlap: int = 50):
        
        self.doc_processor = DocumentProcessor(chunk_size, chunk_overlap)
        self.vector_db = VectorDatabase(embedding_model)
        self.llm_handler = LLMHandler(llm_model)
        self.is_ready = False
        self.llm_model = llm_model
    
   

    def process_document(self,file_path):
        text = ""
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                text += page.extract_text() or ""
        
       

        
        # Chunk document
        chunks = self.doc_processor.chunk_document(text)
        print(f"Created {len(chunks)} chunks")
        
        # Build vector database
        self.vector_db.build_index(chunks)
        
        # Save processed data
        self.doc_processor.save_chunks(chunks, 'chunks/processed_chunks.json')
        self.vector_db.save_index('vectordb/faiss_index.bin', 'vectordb/metadata.pkl')
        
        self.is_ready = True
        print("Document processing complete!")
    
    def load_processed_data(self):
        """Load pre-processed data."""
        try:
            self.vector_db.load_index('vectordb/faiss_index.bin', 'vectordb/metadata.pkl')
            self.is_ready = True
            print("Loaded pre-processed data successfully!")
        except Exception as e:
            print(f"Error loading processed data: {e}")
            self.is_ready = False
    
    def retrieve_context(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve relevant context chunks."""
        if not self.is_ready:
            raise ValueError("Pipeline not ready. Process document first.")
        
        return self.vector_db.search(query, k)
    
    def generate_answer(self, query: str, context_chunks: List[Dict[str, Any]], temperature: float = 0.7) -> str:
        """Generate answer using OpenAI ChatGPT."""
        messages = self.llm_handler.create_prompt(query, context_chunks)
        return self.llm_handler.generate_response(messages, temperature)
    
    def stream_answer(self, query: str, context_chunks: List[Dict[str, Any]], temperature: float = 0.7) -> Generator[str, None, None]:
        """Generate streaming answer using OpenAI ChatGPT."""
        messages = self.llm_handler.create_prompt(query, context_chunks)
        for token in self.llm_handler.stream_response(messages, temperature):
            yield token
    
    def query(self, user_query: str, k: int = 5, temperature: float = 0.7) -> Dict[str, Any]:
        """Complete RAG query pipeline."""
        if not self.is_ready:
            raise ValueError("Pipeline not ready. Process document first.")
        
        # Retrieve context
        context_chunks = self.retrieve_context(user_query, k)
        
        # Generate answer
        answer = self.generate_answer(user_query, context_chunks, temperature)
        
        return {
            'query': user_query,
            'answer': answer,
            'context_chunks': context_chunks,
            'num_chunks_used': len(context_chunks)
        }
    
    def stream_query(self, user_query: str, k: int = 5, temperature: float = 0.7) -> tuple:
        """Stream RAG query pipeline."""
        if not self.is_ready:
            raise ValueError("Pipeline not ready. Process document first.")
        
        # Retrieve context
        context_chunks = self.retrieve_context(user_query, k)
        
        # Return streaming generator and context
        return self.stream_answer(user_query, context_chunks, temperature), context_chunks


