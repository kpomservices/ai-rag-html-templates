import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from typing import List, Optional

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
        self.embeddings = OpenAIEmbeddings(
            model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
        )
        self.llm = ChatOpenAI(
            model=os.getenv("LLM_MODEL", "gpt-3.5-turbo"),
            temperature=0.1
        )
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

Provide a helpful answer about the HTML templates. If the question asks for HTML code, provide clean, well-formatted HTML. Include relevant CSS classes, IDs, and structure information from the templates.

Answer:""",
            input_variables=["context", "question"]
        )
        
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(search_kwargs={"k": 3}),
            chain_type_kwargs={"prompt": qa_prompt},
            return_source_documents=True
        )
        
        # Create HTML generation chain
        html_prompt = PromptTemplate(
            template="""You are an expert HTML developer. Based on the following HTML template examples and user request, generate clean, modern HTML code.

Template examples for reference:
{context}

User request: {question}

Generate complete, valid HTML code that addresses the user's request. Include:
1. Proper HTML5 structure
2. Relevant CSS styling (inline or in <style> tags)
3. Semantic HTML elements
4. Responsive design considerations
5. Clean, readable code structure

Only return the HTML code, no explanations:""",
            input_variables=["context", "question"]
        )
        
        self.html_generation_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(search_kwargs={"k": 5}),
            chain_type_kwargs={"prompt": html_prompt}
        )
    
    def query(self, question: str, max_results: int = 3, include_sources: bool = True):
        """Query the RAG system"""
        if not self.qa_chain:
            raise HTTPException(status_code=500, detail="RAG system not initialized. Run ingest.py first.")
        
        try:
            # Get answer with sources
            result = self.qa_chain({"query": question})
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
                return '\n'.join(html_lines)
            else:
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
    """Query HTML templates and get answers"""
    result = rag_system.query(
        request.query, 
        request.max_results, 
        request.include_sources
    )
    
    return QueryResponse(
        answer=result["answer"],
        sources=result["sources"]
    )

@app.post("/generate", response_model=QueryResponse)
async def generate_html(request: QueryRequest):
    """Generate HTML code based on query"""
    # First get context from templates
    result = rag_system.query(request.query, request.max_results, True)
    
    # Then generate HTML
    html_code = rag_system.generate_html(request.query)
    
    return QueryResponse(
        answer=result["answer"],
        sources=result["sources"] if request.include_sources else None,
        generated_html=html_code
    )

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
    uvicorn.run(app, host="0.0.0.0", port=8000)