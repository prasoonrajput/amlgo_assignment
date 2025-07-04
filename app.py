
import streamlit as st
import os
import time
from src.RAG_pipeline import RAGPipeline
import json


st.set_page_config(
    page_title="RAG Chatbot - Amlgo Labs",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)


st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #2E86AB;
        font-size: 2.5rem;
        margin-bottom: 2rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        display: flex;
        align-items: flex-start;
    }
    .chat-message.user {
        background-color: #E3F2FD;
        flex-direction: row-reverse;
    }
    .chat-message.assistant {
        background-color: #F5F5F5;
    }
    .sidebar-metric {
        background-color: #F0F8FF;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .source-chunk {
        background-color: #FFFACD;
        padding: 0.5rem;
        border-radius: 0.25rem;
        margin: 0.25rem 0;
        border-left: 4px solid #FFD700;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_rag_pipeline():
    """Load and initialize RAG pipeline."""
    pipeline = RAGPipeline(
        embedding_model="all-MiniLM-L6-v2",
        llm_model=os.getenv("LLM_MODEL"),
        # llm_model="gpt-3.5-turbo",
        chunk_size=200,
        chunk_overlap=50
    )
    
    # Try to load pre-processed data
    if os.path.exists('vectordb/faiss_index.bin') and os.path.exists('vectordb/metadata.pkl'):
        pipeline.load_processed_data()
    
    return pipeline

def process_uploaded_file(uploaded_file, pipeline):
    """Process uploaded document."""
    if uploaded_file is not None:
        # Save uploaded file
        os.makedirs('data', exist_ok=True)
        file_path = os.path.join('data', uploaded_file.name)
        
        with open(file_path, 'wb') as f:
            f.write(uploaded_file.getbuffer())
        
        # Process document
        with st.spinner("Processing document... This may take a few minutes."):
            pipeline.process_document(file_path)
        
        st.success("Document processed successfully!")
        return True
    return False

def display_chat_message(role, content):
    """Display chat message with custom styling."""
    with st.chat_message(role):
        st.markdown(content)

def main():
    # Header
    st.markdown('<h1 class="main-header">🤖 RAG Chatbot - Amlgo Labs</h1>', unsafe_allow_html=True)
    
    # Initialize RAG pipeline
    pipeline = load_rag_pipeline()
    
    # Sidebar
    with st.sidebar:
        st.header("📊 System Information")
        
        # System metrics
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Model", "GPT-4o")
            st.metric("Embedding", "MiniLM-L6-v2")
        with col2:
            if pipeline.is_ready:
                num_chunks = len(pipeline.vector_db.chunks)
                st.metric("Chunks", num_chunks)
                st.metric("Status", "Ready ✅")
            else:
                st.metric("Chunks", "0")
                st.metric("Status", "Not Ready ❌")
        
        # API Key check
        if not os.getenv("OPENAI_API_KEY"):
            st.error("⚠️ OpenAI API key not found!")
            st.info("Please set your OPENAI_API_KEY environment variable")
            
        st.divider()
        
        # File upload
        st.header("📄 Document Upload")
        uploaded_file = st.file_uploader(
            "Choose a document file",
            type=['txt', 'pdf', 'docx'],
            help="Upload a document to create a knowledge base"
        )
        
        if uploaded_file is not None and not pipeline.is_ready:
            if st.button("Process Document"):
                if process_uploaded_file(uploaded_file, pipeline):
                    st.rerun()
        
        st.divider()
        
        # Settings
        st.header("⚙️ Settings")
        max_chunks = st.slider("Max Context Chunks", 3, 10, 5)
        temperature = st.slider("Response Temperature", 0.1, 1.0, 0.7, 0.1)
        
        st.divider()
        
        # Clear chat
        if st.button("🗑️ Clear Chat", type="secondary"):
            st.session_state.messages = []
            st.rerun()
        
    
    # Main chat interface
    if not pipeline.is_ready:
        st.warning("⚠️ Please upload and process a document to start chatting!")
        st.info("👈 Use the sidebar to upload a document file")
        
        # Demo section
        st.header("🎯 What this chatbot can do:")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            **📖 Document Analysis**
            - Process large documents
            - Extract key information
            - Understand context
            """)
        
        with col2:
            st.markdown("""
            **🔍 Smart Search**
            - Semantic similarity search
            - Find relevant passages
            - Rank by relevance
            """)
        
        with col3:
            st.markdown("""
            **💬 Natural Conversation**
            - Stream responses in real-time
            - Provide source citations
            - Answer follow-up questions
            """)
        
        return
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat history
    for message in st.session_state.messages:
        display_chat_message(message["role"], message["content"])
        
        # Show sources for assistant messages
        if message["role"] == "assistant" and "sources" in message:
            with st.expander("📚 Source Documents", expanded=False):
                for i, chunk in enumerate(message["sources"]):
                    st.markdown(f"""
                    <div class="source-chunk">
                        <strong>Source {i+1} (Similarity: {chunk.get('similarity_score', 0):.3f})</strong><br>
                        {chunk['text'][:200]}...
                    </div>
                    """, unsafe_allow_html=True)
    
    # Chat input
    if prompt := st.chat_input("Ask a question about the document..."):
        # Add user message to history
        st.session_state.messages.append({"role": "user", "content": prompt})
        display_chat_message("user", prompt)
        
        # Generate assistant response
        with st.chat_message("assistant"):
            # Get streaming response and context
            try:
                response_generator, context_chunks = pipeline.stream_query(prompt, max_chunks, temperature)
                
                # Create placeholder for streaming response
                response_placeholder = st.empty()
                full_response = ""
                
                # Stream the response
                for token in response_generator:
                    full_response += token
                    response_placeholder.markdown(full_response + "▌")
                    time.sleep(0.02)  # Faster streaming for better ChatGPT experience
                
                # Final response without cursor
                response_placeholder.markdown(full_response)
                
                # Show source documents
                if context_chunks:
                    with st.expander("📚 Source Documents", expanded=False):
                        for i, chunk in enumerate(context_chunks):
                            st.markdown(f"""
                            <div class="source-chunk">
                                <strong>Source {i+1} (Similarity: {chunk.get('similarity_score', 0):.3f})</strong><br>
                                <small>Chunk {chunk['chunk_index']} | {chunk['word_count']} words</small><br>
                                {chunk['text']}
                            </div>
                            """, unsafe_allow_html=True)
                
                # Add assistant message to history
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": full_response,
                    "sources": context_chunks
                })
                
            except Exception as e:
                st.error(f"Error generating response: {str(e)}")
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": "I apologize, but I encountered an error while processing your question. Please try again."
                })

if __name__ == "__main__":
    main()



