# RAG Chatbot with ChatGPT - Amlgo Labs Assignment

## Overview

This is a Retrieval-Augmented Generation (RAG) chatbot built with Streamlit that uses OpenAI's ChatGPT to answer questions based on uploaded documents. The system uses semantic search to find relevant document chunks and generates contextual answers using GPT-3.5-turbo.

## Features

- 📄 Document upload and processing (TXT, PDF, DOCX)
- 🔍 Semantic search using sentence transformers
- 💬 Real-time streaming responses with ChatGPT
- 📚 Source document attribution
- 🎯 Interactive Streamlit interface
- ⚙️ Configurable parameters (temperature, context chunks)

## Architecture

1. **Document Processing**: Clean, chunk, and embed documents
2. **Vector Database**: FAISS for efficient similarity search
3. **Language Model**: OpenAI GPT-3.5-turbo for response generation
4. **Streamlit UI**: Interactive chat interface with streaming

## Installation

1. Clone the repository:

```bash
git clone https://github.com/prasoonrajput/amlgo_assignment.git
cd amlgo_assignment
```

2. Create virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Set up OpenAI API key:

```bash
# Create a .env file in the project root
echo "OPENAI_API_KEY=your_api_key_here" > .env

# Or set environment variable directly
export OPENAI_API_KEY="your_api_key_here"
```

5. Run setup:

```bash
python setup.py
```

## Usage

1. Get your OpenAI API key from [OpenAI Platform](https://platform.openai.com/api-keys)

2. Set your API key (choose one method):

   - Create `.env` file: `OPENAI_API_KEY=your_key_here`
   - Set environment variable: `export OPENAI_API_KEY="your_key_here"`
   - Use Streamlit secrets: Add to `.streamlit/secrets.toml`

3. Start the Streamlit app:

```bash
streamlit run app.py
```

4. Upload a document using the sidebar
5. Wait for processing to complete
6. Start chatting with your document!

## Sample Queries

- "What are the main terms and conditions?"
- "How is user data protected?"
- "What are the refund policies?"
- "Who can I contact for support?"
- "What are the privacy implications?"

## Technical Details

### Document Processing

- Chunks: 200 words with 50-word overlap
- Embedding: all-MiniLM-L6-v2 (384 dimensions)
- Vector DB: FAISS with L2 distance

### Language Model

- Model: OpenAI GPT-4o
- Temperature: Configurable (0.1-1.0)
- Max tokens: 512
- Streaming: Real-time token streaming

### Performance

- Streaming: Real-time token generation from OpenAI
- Search: Sub-second semantic search
- Cost: ~$0.002 per 1K tokens (input + output)

## Customization

- Modify chunk size in `DocumentProcessor`
- Change embedding model in `VectorDatabase`
- Adjust ChatGPT parameters in `LLMHandler`
- Switch to GPT-4 for better quality (higher cost)
- Customize UI in `app.py`

## Troubleshooting

### API Key Issues

- Ensure OpenAI API key is set correctly
- Check API key has sufficient credits
- Verify API key permissions

### Performance Issues

- Reduce chunk size for faster processing
- Use fewer context chunks for faster responses
- Consider using text-embedding-ada-002 for better embeddings

### Common Errors

- **"API key not found"**: Set OPENAI_API_KEY environment variable
- **"Rate limit exceeded"**: Wait or upgrade OpenAI plan
- **"Context too long"**: Reduce chunk size or number of chunks

## Advanced Features

- **Model Selection**: Switch between GPT-3.5-turbo and GPT-4
- **Temperature Control**: Adjust response creativity
- **Token Counting**: Monitor API usage
- **Context Window**: Smart context management

## License

This project is for educational purposes as part of the Amlgo Labs assignment.
"""
