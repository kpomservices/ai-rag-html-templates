import os
import traceback
import logging
import torch
import asyncio

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline, AutoTokenizer
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from typing import List, Optional
from langchain_community.llms import Ollama
# from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

load_dotenv()

app = FastAPI(title="HTML Template RAG API", version="1.0.0")

class QueryRequest(BaseModel):
    query: str
    max_results: Optional[int] = 3
    include_sources: Optional[bool] = True

class QueryResponse(BaseModel):
    answer: str
    sources: Optional[List[dict]] = None
    generated_html: Optional[str] = None

class HTMLRAGSystem:
    def __init__(self, persist_directory="./chroma_db"):
        self.persist_directory = persist_directory
        # self.embeddings = OpenAIEmbeddings(
        #     model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
        # )
        self.embeddings = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')

        # "model" => "gpt-4o-mini",
        # self.llm = ChatOpenAI(
        #     # model=os.getenv("LLM_MODEL", "gpt-3.5-turbo"),
        #     model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
        #     temperature=0.1
        # )

        # Get model from environment or use default
        # model_name = os.getenv("LLM_MODEL", "microsoft/DialoGPT-medium")
        # model_name = 'gpt2';        
        
        # Fixed configuration
        # model_name = os.getenv("LLM_MODEL", "distilgpt2")
 
        # Try Ollama first, fallback to HuggingFace if not available
        try:
            self.llm = Ollama(
                model="tinyllama",  # Fast and reliable model
                temperature=0.1,
                num_ctx=4096,      # Increased context window
                num_predict=2048,  # Increased max tokens to generate for longer responses
                stop=["</s>", "Human:", "Assistant:"]  # Stop tokens to prevent incomplete responses
            )
            print("✅ Using Ollama with tinyllama model")
        except Exception as e:
            print(f"⚠️ Ollama not available: {e}")
            print("🔄 Falling back to HuggingFace model...")
            
            # Fallback to HuggingFace model
            model_name = "distilgpt2"
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
                
            pipe = pipeline(
                "text-generation",
                model=model_name,
                tokenizer=tokenizer,
                max_new_tokens=1024,  # Increased from 150 to 1024 for longer responses
                temperature=0.1,
                do_sample=True,
                pad_token_id=50256,
                device=0 if torch.cuda.is_available() else -1,
                return_full_text=False,
                eos_token_id=tokenizer.eos_token_id,  # Explicit end-of-sequence token
                repetition_penalty=1.1  # Prevent repetitive output
            )

            self.llm = HuggingFacePipeline(
                pipeline=pipe,
                pipeline_kwargs={
                    "max_new_tokens": 1024,  # Increased from 150 to 1024 for longer responses
                    "temperature": 0.1,
                    "do_sample": True,
                    "return_full_text": False,
                    "eos_token_id": tokenizer.eos_token_id,  # Explicit end-of-sequence token
                    "repetition_penalty": 1.1  # Prevent repetitive output
                }
            )
            print("✅ Using HuggingFace distilgpt2 model")
        # Initialize with explicit tokenizer
        # tokenizer = AutoTokenizer.from_pretrained(model_name)
        # if tokenizer.pad_token is None:
        #     tokenizer.pad_token = tokenizer.eos_token
            
        # pipe = pipeline(
        #     "text-generation",
        #     model=model_name,
        #     tokenizer=tokenizer,
        #     max_new_tokens=150,        # Use max_new_tokens instead of max_length
        #     temperature=0.1,
        #     do_sample=True,
        #     pad_token_id=50256,
        #     device=0 if torch.cuda.is_available() else -1,
        #     return_full_text=False     # Only return the generated part
        # )

        # self.llm = HuggingFacePipeline(
        #     pipeline=pipe,
        #     model_kwargs={
        #         "max_new_tokens": 150,    # New tokens to generate
        #         "temperature": 0.1,
        #         "do_sample": True,
        #         "return_full_text": False
        #     }
        # )

        # # Create the pipeline
        # pipe = pipeline(
        #     "text-generation",
        #     model=model_name,
        #     max_length=512,
        #     temperature=0.1,  # Same temperature as your OpenAI config
        #     do_sample=True,
        #     pad_token_id=50256,  # Helps avoid warnings
        #     device=0 if torch.cuda.is_available() else -1
        # )
        
        # self.llm = HuggingFacePipeline(
        #     pipeline=pipe,
        #     model_kwargs={
        #         "temperature": 0.1,
        #         "max_length": 512,
        #         "do_sample": True,
        #     }
        # )
    
        self.vectorstore = None
        self.qa_chain = None
        self.html_generation_chain = None
        self._initialize()
    
    def _initialize(self):
        """Initialize the RAG system"""
        try:
            # Load existing vector store
            self.vectorstore = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings
            )
            
            # Test if vector store has documents
            test_results = self.vectorstore.similarity_search("test", k=1)
            if not test_results:
                raise Exception("No documents in vector store")
                
        except Exception as e:
            print(f"Vector store not found or empty: {e}")
            print("Please run ingest.py first to create the vector database")
            return
        
        # Create QA chain for answering questions
        qa_prompt = PromptTemplate(
            template="""You are an expert HTML template assistant. Use the following HTML template pieces to answer questions about HTML templates, components, and web development.

Context from HTML templates:
{context}

Question: {question}

Provide a comprehensive and detailed answer about the HTML templates. If the question asks for HTML code, provide complete, clean, well-formatted HTML with all necessary elements. Include relevant CSS classes, IDs, and structure information from the templates. Be thorough and provide complete solutions.

Answer:""",
            input_variables=["context", "question"]
        )
        
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(search_kwargs={"k": 3}),
            chain_type_kwargs={
                "prompt": qa_prompt,
                "document_separator": "\n\n",  # Better document separation
                "document_variable_name": "context"  # Explicit variable name
            },
            return_source_documents=True
        )
        
        # Create HTML generation chain
        html_prompt = PromptTemplate(
            template="""You are an expert HTML developer. Based on the following HTML template examples and user request, generate complete, clean, modern HTML code.

Template examples for reference:
{context}

User request: {question}

Generate comprehensive, valid HTML code that fully addresses the user's request. Include:
1. Complete HTML5 structure with proper DOCTYPE and meta tags
2. Comprehensive CSS styling (inline or in <style> tags)
3. Semantic HTML elements for accessibility
4. Responsive design considerations with media queries
5. Clean, readable code structure with proper indentation
6. All necessary JavaScript if required
7. Complete implementation of the requested functionality

Provide the complete HTML code without any explanations or additional text:""",
            input_variables=["context", "question"]
        )
        
        self.html_generation_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(search_kwargs={"k": 5}),
            chain_type_kwargs={
                "prompt": html_prompt,
                "document_separator": "\n\n",  # Better document separation
                "document_variable_name": "context"  # Explicit variable name
            }
        )
    
    def query(self, question: str, max_results: int = 3, include_sources: bool = True):
        """Query the RAG system"""
        if not self.qa_chain:
            raise HTTPException(status_code=500, detail="RAG system not initialized. Run ingest.py first.")
        
        try:
            # Get answer with sources
            result = self.qa_chain.invoke({"query": question})
            answer = result["result"]
            
            # Log the answer length for debugging
            logger.info(f"Generated answer length: {len(answer)} characters")
            logger.info(f"Answer preview: {answer[:200]}...")
            
            sources = []
            if include_sources and "source_documents" in result:
                for doc in result["source_documents"][:max_results]:
                    sources.append({
                        "filename": doc.metadata.get("filename", "Unknown"),
                        "source": doc.metadata.get("source", "Unknown"),
                        "title": doc.metadata.get("title", ""),
                        "content_preview": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
                    })
            
            return {
                "answer": answer,
                "sources": sources if include_sources else None
            }
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")
    
    def generate_html(self, request: str):
        """Generate HTML code based on request"""
        if not self.html_generation_chain:
            raise HTTPException(status_code=500, detail="HTML generation system not initialized. Run ingest.py first.")
        
        try:
            result = self.html_generation_chain({"query": request})
            html_code = result["result"]
            
            # Log the HTML code length for debugging
            logger.info(f"Generated HTML length: {len(html_code)} characters")
            logger.info(f"HTML preview: {html_code[:200]}...")
            
            # Clean up the HTML code (remove any explanatory text)
            lines = html_code.split('\n')
            html_lines = []
            in_html = False
            
            for line in lines:
                if '<!DOCTYPE' in line or '<html' in line:
                    in_html = True
                if in_html:
                    html_lines.append(line)
                if '</html>' in line:
                    break
            
            if html_lines:
                final_html = '\n'.join(html_lines)
                logger.info(f"Final HTML length after cleanup: {len(final_html)} characters")
                return final_html
            else:
                logger.info(f"Using original HTML code, length: {len(html_code)} characters")
                return html_code
                
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error generating HTML: {str(e)}")

# Initialize the RAG system
rag_system = HTMLRAGSystem()

@app.get("/")
async def root():
    return {"message": "HTML Template RAG API", "status": "running"}

@app.post("/query", response_model=QueryResponse)
async def query_templates(request: QueryRequest):

    try:
        logger.info(f"Received query: {request.query}")
        
        logger.info("About to invoke rag_system")
                
        """Query HTML templates and get answers"""
        
        # Server-side timeout handling
        result = await asyncio.wait_for(
            asyncio.to_thread(rag_system.query, request.query, request.max_results, request.include_sources),
            timeout=480000  # 20 minutes
        )
        
        # result = rag_system.query(
        #     request.query, 
        #     request.max_results, 
        #     request.include_sources
        # )
        logger.info(f"Query HTML templates result: {result}")
        logger.info(f"Answer length in response: {len(result['answer'])} characters")
        
        response = QueryResponse(
            answer=result["answer"],
            sources=result["sources"]
        )
        
        logger.info(f"Final response answer length: {len(response.answer)} characters")
        return response
        
    except Exception as e:
        error_msg = f"Error in query endpoint: {str(e)}"
        logger.error(error_msg)
        logger.error(f"Full traceback: {traceback.format_exc()}")
        
        # Print to console as well
        print(f"ERROR: {error_msg}")
        print(f"TRACEBACK: {traceback.format_exc()}")
        
        raise HTTPException(status_code=500, detail=error_msg)
        

@app.post("/generate", response_model=QueryResponse)
async def generate_html(request: QueryRequest):
    """Generate HTML code based on query"""
      
    try:
        logger.info(f"Received query: {request.query}")
        
        logger.info("About to invoke rag_system")
                
        """Query HTML templates and get answers"""
        # First get context from templates
        # result = rag_system.query(request.query, request.max_results, True)
        
        # Server-side timeout handling
        result = await asyncio.wait_for(
            asyncio.to_thread(rag_system.query, request.query, request.max_results, True),
            timeout=480000  # 20 minutes
        )

        # Then generate HTML
        html_code = rag_system.generate_html(request.query)
        
        logger.info(f"Generated HTML length: {len(html_code)} characters")
        
        response = QueryResponse(
            answer=result["answer"],
            sources=result["sources"] if request.include_sources else None,
            generated_html=html_code
        )
        
        logger.info(f"Final response answer length: {len(response.answer)} characters")
        logger.info(f"Final response HTML length: {len(response.generated_html) if response.generated_html else 0} characters")
        return response
        
    except Exception as e:
        error_msg = f"Error in query endpoint: {str(e)}"
        logger.error(error_msg)
        logger.error(f"Full traceback: {traceback.format_exc()}")
        
        # Print to console as well
        print(f"ERROR: {error_msg}")
        print(f"TRACEBACK: {traceback.format_exc()}")
        
        raise HTTPException(status_code=500, detail=error_msg)
        

   

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Test vector store
        if rag_system.vectorstore:
            test_results = rag_system.vectorstore.similarity_search("test", k=1)
            return {
                "status": "healthy",
                "vector_store": "connected",
                "documents_available": len(test_results) > 0
            }
        else:
            return {
                "status": "unhealthy",
                "vector_store": "not initialized",
                "message": "Run ingest.py first"
            }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, timeout_keep_alive=120)