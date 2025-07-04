
import openai
import os
from typing import Generator, List, Dict, Any
import tiktoken
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class LLMHandler:
    def __init__(self, model_name):
        self.model_name = model_name
        self.api_key = os.getenv("OPENAI_API_KEY")
        
        if os.getenv("BASE_PATH"):
            base_url=os.getenv("BASE_PATH")
        
        if not self.api_key:
            raise ValueError("OpenAI API key not found. Please set OPENAI_API_KEY environment variable.")
        
        # Initialize OpenAI client
        self.client = openai.OpenAI(api_key=self.api_key,base_url=base_url)
        
        # Initialize tokenizer for token counting
        try:
            self.tokenizer = tiktoken.encoding_for_model(model_name)
        except KeyError:
            # Fallback to cl100k_base for newer models
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
        
        print(f"Initialized OpenAI client with model: {self.model_name}")
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        return len(self.tokenizer.encode(text))
    
    def create_prompt(self, query: str, context_chunks: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """Create a chat prompt with context and query."""
        context_text = "\n\n".join([chunk['text'] for chunk in context_chunks])
        
        # Create messages for ChatGPT
        messages = [
            {
                "role": "system",
                "content": """You are a helpful assistant that answers questions based on provided context documents. 
                
Rules:
1. Only use information from the provided context to answer questions
2. If the answer cannot be found in the context, say "I don't have enough information to answer this question based on the provided documents."
3. Be accurate, concise, and helpful
4. Quote relevant parts of the context when appropriate
5. If the context is unclear or contradictory, mention this in your response"""
            },
            {
                "role": "user", 
                "content": f"""Context Documents:
{context_text}

Question: {query}

Please answer the question based on the context provided above."""
            }
        ]
        
        return messages
    
    def generate_response(self, messages: List[Dict[str, str]], temperature: float = 0.7, max_tokens: int = 512) -> str:
        """Generate a complete response."""
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=1.0,
                frequency_penalty=0.0,
                presence_penalty=0.0
            )
            
            return response.choices[0].message.content.strip()
        
        except Exception as e:
            print(f"Error generating response: {e}")
            return "I apologize, but I encountered an error while generating a response. Please try again."
    
    def stream_response(self, messages: List[Dict[str, str]], temperature: float = 0.7, max_tokens: int = 512) -> Generator[str, None, None]:
        """Generate streaming response."""
        try:
            stream = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=1.0,
                frequency_penalty=0.0,
                presence_penalty=0.0,
                stream=True
            )
            
            for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    yield chunk.choices[0].delta.content
        
        except Exception as e:
            print(f"Error streaming response: {e}")
            yield "I apologize, but I encountered an error while generating a response. Please try again."


