import os
import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_community.llms import GPT4All
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from typing import List, Optional
import uvicorn
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

app = FastAPI(title="HTML Template RAG API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    query: str
    max_results: Optional[int] = 3
    include_sources: Optional[bool] = True

class QueryResponse(BaseModel):
    answer: str
    sources: Optional[List[dict]] = None
    generated_html: Optional[str] = None
    processing_time: Optional[float] = None
    mode: Optional[str] = "ai"  # "ai" or "fallback"

class HTMLRAGSystem:
    def __init__(self, persist_directory="./chroma_db"):
        self.persist_directory = persist_directory
        self.model_path = os.getenv("GGUF_MODEL_PATH", "models/q4_0-orca-mini-3b.gguf")
        self.embeddings = None
        self.llm = None
        self.vectorstore = None
        self.qa_chain = None
        self.html_generation_chain = None
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.model_loaded = False
        self.vectorstore_loaded = False
        self._initialize()
    
    def _initialize(self):
        """Initialize the RAG system with better error handling"""
        try:
            logger.info("Starting RAG system initialization...")
            
            # First, try to load vector store (this is essential)
            self._load_vectorstore()
            
            # Then try to load the AI model (this is optional for fallback mode)
            if self.vectorstore_loaded:
                self._load_model()
            
        except Exception as e:
            logger.error(f"RAG system initialization failed: {str(e)}")
    
    def _load_vectorstore(self):
        """Load vector store"""
        try:
            logger.info("Loading vector store...")
            self.embeddings = GPT4AllEmbeddings(
                model_path=self.model_path,
                n_threads=4,
            )
            
            self.vectorstore = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings
            )
            
            # Test vector store
            test_results = self.vectorstore.similarity_search("test", k=1)
            if not test_results:
                logger.error("No documents in vector store")
                return
            
            self.vectorstore_loaded = True
            logger.info(f"Vector store loaded successfully with {len(test_results)} documents")
            
        except Exception as e:
            logger.error(f"Failed to load vector store: {str(e)}")
    
    def _load_model(self):
        """Load AI model"""
        try:
            logger.info("Loading AI model...")
            
            if not os.path.exists(self.model_path):
                logger.error(f"Model file not found: {self.model_path}")
                return
            
            # Initialize LLM with conservative settings
            self.llm = GPT4All(
                model=self.model_path,
                n_threads=2,  # Very conservative
                verbose=False,
            )
            
            # Create chains with updated LangChain API
            self._create_chains()
            self.model_loaded = True
            logger.info("AI model loaded successfully!")
            
        except Exception as e:
            logger.error(f"Failed to load AI model: {str(e)}")
            logger.info("System will operate in fallback mode using vector search only")
    
    def _create_chains(self):
        """Create the QA chains using new LangChain API"""
        qa_prompt = PromptTemplate(
            template="""Use the following HTML template pieces to answer the question concisely.

Context: {context}

Question: {question}

Answer briefly:""",
            input_variables=["context", "question"]
        )
        
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(search_kwargs={"k": 2}),
            chain_type_kwargs={"prompt": qa_prompt},
            return_source_documents=True
        )
        
        html_prompt = PromptTemplate(
            template="""Generate simple HTML code for: {question}

Reference templates: {context}

HTML only:""",
            input_variables=["context", "question"]
        )
        
        self.html_generation_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(search_kwargs={"k": 2}),
            chain_type_kwargs={"prompt": html_prompt}
        )
    
    def _vector_search_fallback(self, question: str, max_results: int = 3):
        """Fallback using only vector search when AI model is unavailable"""
        if not self.vectorstore_loaded:
            return {
                "answer": "Vector database is not available. Please check system initialization.",
                "sources": [],
                "mode": "error"
            }
        
        try:
            # Perform similarity search
            docs = self.vectorstore.similarity_search(question, k=max_results)
            
            if not docs:
                return {
                    "answer": f"No relevant templates found for '{question}'. Try a different search term.",
                    "sources": [],
                    "mode": "fallback"
                }
            
            # Build sources from search results
            sources = []
            relevant_content = []
            
            for doc in docs:
                sources.append({
                    "filename": doc.metadata.get("filename", "Unknown"),
                    "source": doc.metadata.get("source", "Unknown"),
                    "title": doc.metadata.get("title", ""),
                    "content_preview": doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content
                })
                relevant_content.append(doc.page_content[:200])
            
            # Create a simple answer from the search results
            answer = f"Found {len(docs)} relevant template(s) for '{question}'. "
            answer += "Review the source documents below for detailed information. "
            answer += "(AI model unavailable - showing search results only)"
            
            return {
                "answer": answer,
                "sources": sources,
                "mode": "fallback"
            }
            
        except Exception as e:
            logger.error(f"Vector search fallback failed: {e}")
            return {
                "answer": f"Search failed: {str(e)}",
                "sources": [],
                "mode": "error"
            }
    
    def _run_query_with_timeout(self, question: str, max_results: int = 3, include_sources: bool = True, timeout: int = 25):
        """Run query with strict timeout"""
        start_time = time.time()
        
        if not self.model_loaded or not self.qa_chain:
            result = self._vector_search_fallback(question, max_results)
            result["processing_time"] = time.time() - start_time
            return result
        
        try:
            logger.info(f"Processing AI query: {question[:50]}...")
            
            # Use the new invoke method instead of __call__
            result = self.qa_chain.invoke({"query": question})
            answer = result["result"]
            
            sources = []
            if include_sources and "source_documents" in result:
                for doc in result["source_documents"][:max_results]:
                    sources.append({
                        "filename": doc.metadata.get("filename", "Unknown"),
                        "source": doc.metadata.get("source", "Unknown"),
                        "title": doc.metadata.get("title", ""),
                        "content_preview": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
                    })
            
            processing_time = time.time() - start_time
            logger.info(f"AI query completed in {processing_time:.2f} seconds")
            
            return {
                "answer": answer,
                "sources": sources if include_sources else None,
                "processing_time": processing_time,
                "mode": "ai"
            }
            
        except Exception as e:
            logger.error(f"AI query failed: {str(e)}")
            # Fall back to vector search
            result = self._vector_search_fallback(question, max_results)
            result["processing_time"] = time.time() - start_time
            return result
    
    async def query(self, question: str, max_results: int = 3, include_sources: bool = True):
        """Async query with timeout handling"""
        try:
            # Run with strict timeout
            future = self.executor.submit(
                self._run_query_with_timeout,
                question,
                max_results,
                include_sources,
                25  # 25 second timeout for processing
            )
            
            result = await asyncio.wait_for(
                asyncio.wrap_future(future),
                timeout=30.0  # 30 second total timeout
            )
            
            return result
            
        except (asyncio.TimeoutError, FutureTimeoutError):
            logger.error("Query timed out, using fallback")
            return self._vector_search_fallback(question, max_results)
        except Exception as e:
            logger.error(f"Async query error: {str(e)}")
            return self._vector_search_fallback(question, max_results)
    
    async def generate_html(self, request: str):
        """Generate HTML with fallback"""
        if not self.model_loaded:
            # Simple HTML generation fallback
            return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{request}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
        .container {{ max-width: 800px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; }}
        .header {{ color: #333; border-bottom: 2px solid #007bff; padding-bottom: 10px; }}
    </style>
</head>
<body>
    <div class="container">
        <h1 class="header">{request}</h1>
        <p>This is a basic HTML template generated for your request.</p>
        <p><em>AI model unavailable - using template fallback</em></p>
    </div>
</body>
</html>"""
        
        try:
            future = self.executor.submit(
                lambda: self.html_generation_chain.invoke({"query": request})["result"]
            )
            
            result = await asyncio.wait_for(
                asyncio.wrap_future(future),
                timeout=30.0
            )
            
            return result
            
        except Exception as e:
            logger.error(f"HTML generation failed: {e}")
            return self.generate_html(request)  # Use fallback

# Initialize system
logger.info("Starting HTML Template RAG API...")
rag_system = HTMLRAGSystem()

@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response

@app.get("/")
async def root():
    return {
        "message": "HTML Template RAG API", 
        "status": "running",
        "model_loaded": rag_system.model_loaded if rag_system else False,
        "vectorstore_loaded": rag_system.vectorstore_loaded if rag_system else False,
        "mode": "ai" if (rag_system and rag_system.model_loaded) else "fallback"
    }

@app.post("/query", response_model=QueryResponse)
async def query_templates(request: QueryRequest):
    """Query HTML templates and get answers"""
    if not rag_system:
        raise HTTPException(status_code=503, detail="RAG system not initialized")
    
    try:
        result = await rag_system.query(
            request.query, 
            request.max_results, 
            request.include_sources
        )
        
        return QueryResponse(
            answer=result["answer"],
            sources=result["sources"],
            processing_time=result.get("processing_time"),
            mode=result.get("mode", "unknown")
        )
        
    except Exception as e:
        logger.error(f"API Error: {str(e)}")
        return QueryResponse(
            answer=f"Error processing query: {str(e)}",
            sources=[],
            mode="error"
        )

@app.post("/generate", response_model=QueryResponse)
async def generate_html(request: QueryRequest):
    """Generate HTML code based on query"""
    if not rag_system:
        raise HTTPException(status_code=503, detail="RAG system not initialized")
    
    try:
        # Get context first
        query_result = await rag_system.query(request.query, request.max_results, True)
        
        # Generate HTML
        html_code = await rag_system.generate_html(request.query)
        
        return QueryResponse(
            answer=query_result["answer"],
            sources=query_result["sources"] if request.include_sources else None,
            generated_html=html_code,
            processing_time=query_result.get("processing_time"),
            mode=query_result.get("mode", "unknown")
        )
        
    except Exception as e:
        logger.error(f"Generate API Error: {str(e)}")
        return QueryResponse(
            answer=f"Error generating HTML: {str(e)}",
            sources=[],
            mode="error"
        )

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    if not rag_system:
        return {"status": "error", "message": "RAG system not initialized"}
    
    return {
        "status": "healthy" if rag_system.vectorstore_loaded else "error",
        "model_loaded": rag_system.model_loaded,
        "vectorstore_loaded": rag_system.vectorstore_loaded,
        "model_path": rag_system.model_path,
        "model_exists": os.path.exists(rag_system.model_path) if rag_system.model_path else False,
        "mode": "ai" if rag_system.model_loaded else "fallback"
    }

if __name__ == "__main__":
    logger.info("Starting server on http://localhost:8000")
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,
        timeout_keep_alive=120,
        timeout_graceful_shutdown=60
    )