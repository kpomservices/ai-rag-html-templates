import os
import glob
from pathlib import Path
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import GPT4AllEmbeddings
# from langchain_community.llms import GPT4All
from langchain import hub
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
from dotenv import load_dotenv

load_dotenv()

class HTMLTemplateIngester:
    def __init__(self, persist_directory="./chroma_db"):
        self.persist_directory = persist_directory

        model_path = os.getenv("GGUF_MODEL_PATH", "models/q4_0-orca-mini-3b.gguf")
        self.embeddings = GPT4AllEmbeddings(
            model_path=model_path,
            n_ctx=2048,
            n_threads=6,
            n_batch=64,
        )

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=int(os.getenv("CHUNK_SIZE", 1000)),
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP", 200)),
            separators=["\n\n", "\n", " ", ""]
        )
        
    def extract_html_info(self, file_path):
        """Extract meaningful information from HTML files"""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        soup = BeautifulSoup(content, 'html.parser')
        
        # Extract metadata
        title = soup.find('title')
        title_text = title.get_text().strip() if title else ""
        
        # Extract meta description
        meta_desc = soup.find('meta', attrs={'name': 'description'})
        description = meta_desc.get('content', '') if meta_desc else ""
        
        # Extract main content areas
        sections = []
        
        # Headers
        for header in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
            sections.append(f"Header: {header.get_text().strip()}")
        
        # Extract classes and IDs for styling reference
        elements_with_classes = soup.find_all(class_=True)
        classes = set()
        for elem in elements_with_classes[:10]:  # Limit to avoid too much noise
            classes.update(elem.get('class', []))
        
        elements_with_ids = soup.find_all(id=True)
        ids = [elem.get('id') for elem in elements_with_ids[:10]]
        
        # Create structured content
        structured_content = f"""
File: {Path(file_path).name}
Title: {title_text}
Description: {description}
Classes: {', '.join(list(classes)[:10])}
IDs: {', '.join(ids)}
Headers: {'; '.join(sections)}

Raw HTML:
{content}
        """.strip()
        
        return structured_content, {
            'source': str(file_path),
            'filename': Path(file_path).name,
            'title': title_text,
            'description': description,
            'type': 'html_template'
        }
    
    def ingest_templates(self, templates_dir="./templates"):
        """Ingest all HTML templates from directory"""
        documents = []
        
        # Support multiple file extensions
        patterns = ["*.html", "*.htm", "*.mjml"]
        
        for pattern in patterns:
            for file_path in glob.glob(os.path.join(templates_dir, "**", pattern), recursive=True):
                print(f"Processing: {file_path}")
                
                try:
                    content, metadata = self.extract_html_info(file_path)
                    
                    # Split content into chunks
                    texts = self.text_splitter.split_text(content)
                    
                    # Create documents
                    for i, text in enumerate(texts):
                        doc_metadata = metadata.copy()
                        doc_metadata['chunk_id'] = i
                        doc_metadata['total_chunks'] = len(texts)
                        
                        documents.append(Document(
                            page_content=text,
                            metadata=doc_metadata
                        ))
                        
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")
        
        if not documents:
            print("No documents found. Creating sample templates...")
            self.create_sample_templates(templates_dir)
            return self.ingest_templates(templates_dir)
        
        print(f"Creating vector store with {len(documents)} document chunks...")
        
        # Create vector store
        vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            persist_directory=self.persist_directory
        )
        
        print(f"Successfully ingested {len(documents)} chunks from HTML templates")
        return vectorstore

if __name__ == "__main__":
    ingester = HTMLTemplateIngester()
    vectorstore = ingester.ingest_templates()
