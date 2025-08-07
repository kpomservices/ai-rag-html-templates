import os
import glob
from pathlib import Path
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
from dotenv import load_dotenv

# Option 1: Use langchain_huggingface (recommended)
try:
    from langchain_huggingface import HuggingFaceEmbeddings
    HUGGINGFACE_AVAILABLE = True
except ImportError:
    HUGGINGFACE_AVAILABLE = False
    print("Warning: langchain_huggingface not found. Using alternative embedding method.")

# Option 2: Alternative imports if langchain_huggingface is not available
if not HUGGINGFACE_AVAILABLE:
    try:
        from langchain_community.embeddings import HuggingFaceEmbeddings
    except ImportError:
        try:
            from langchain.embeddings import HuggingFaceEmbeddings
        except ImportError:
            print("Error: No HuggingFace embeddings package found. Please install one of:")
            print("- pip install langchain-huggingface")
            print("- pip install langchain-community")
            raise ImportError("No suitable embedding package found")

load_dotenv()

class HTMLTemplateIngester:
    def __init__(self, persist_directory="./chroma_db"):
        self.persist_directory = persist_directory
        self.embeddings = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=int(os.getenv("CHUNK_SIZE", 2000)),  # Increased from 1000
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP", 300)),  # Increased from 200  
            separators=["\n\n", "\n", " ", ""],
            length_function=len,
            keep_separator=True  # Keep separators to maintain structure
        )
    
    def debug_template_content(self, templates_dir, filename=None):
        """Debug method to inspect template content and chunking"""
        if filename:
            # Debug specific file
            file_path = os.path.join(templates_dir, filename)
            if os.path.exists(file_path):
                print(f"\n=== DEBUGGING: {filename} ===")
                content, metadata = self.extract_html_info(file_path)
                
                print(f"Original content length: {len(content)} characters")
                print(f"Metadata: {metadata}")
                
                # Show first 500 characters
                print(f"\nFirst 500 characters of structured content:")
                print("-" * 50)
                print(content[:500])
                print("-" * 50)
                
                # Test chunking
                texts = self.text_splitter.split_text(content)
                print(f"\nChunked into {len(texts)} pieces:")
                for i, chunk in enumerate(texts):
                    print(f"Chunk {i+1}: {len(chunk)} characters")
                    if i == 0:  # Show first chunk
                        print("First chunk preview:")
                        print(chunk[:200] + "..." if len(chunk) > 200 else chunk)
                        print()
            else:
                print(f"File {filename} not found in {templates_dir}")
        else:
            # Debug all files
            templates = self.create_sample_templates_from_folder(templates_dir)
            for fname in templates.keys():
                self.debug_template_content(templates_dir, fname)
                print("\n" + "="*60 + "\n")

    def create_sample_templates_from_folder(self, templates_dir):
        """Read all HTML templates from a directory and return a dictionary of {filename: content}"""
        if not os.path.exists(templates_dir):
            print(f"Directory '{templates_dir}' does not exist. Creating it...")
            os.makedirs(templates_dir, exist_ok=True)
            return {}
        
        templates = {}
        
        for filename in os.listdir(templates_dir):
            file_path = os.path.join(templates_dir, filename)
            # FIXED: Now includes .txt files like the ingest_templates method
            if filename.endswith((".html", ".htm", ".mjml", ".txt")) and os.path.isfile(file_path):
                try:
                    with open(file_path, 'r', encoding='utf-8') as file:
                        templates[filename] = file.read()
                        print(f"Loaded template: {filename}")
                except Exception as e:
                    print(f"Error reading {filename}: {e}")
        
        print(f"Found {len(templates)} templates in {templates_dir}")
        return templates
        
    def extract_html_info(self, file_path):
        """Extract meaningful information from HTML files"""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check if it's a text file or HTML file
        file_extension = Path(file_path).suffix.lower()
        
        if file_extension == '.txt':

             # Check if MJML tags exist inside the .txt file
            if "<mjml" in content and "</mjml>" in content:
                print(f"Detected MJML content inside .txt file: {file_path}")
                soup = BeautifulSoup(content, 'xml')

                mjml_texts = [tag.get_text(strip=True) for tag in soup.find_all('mj-text')]
                mjml_buttons = [tag.get_text(strip=True) for tag in soup.find_all('mj-button')]
                mjml_images = [tag.get('src') for tag in soup.find_all('mj-image') if tag.get('src')]
            print(f"❌ MJML ============== TEXT templates found")
            print(f"+===========+ MJML '{soup}' +===========+")
            # For text files, create simpler metadata
#             structured_content = f"""
# === TEMPLATE METADATA ===
# Type: MJML Template (from .txt)
# Text Blocks: {len(mjml_texts)}
# Buttons: {', '.join(mjml_buttons) if mjml_buttons else 'None'}
# Images: {', '.join(mjml_images) if mjml_images else 'None'}
# Content Length: {len(content)} characters

# === TEXT CONTENT ===
# {content}

# === END OF TEMPLATE ===
#             """.strip()
            
            return structured_content, {
                'source': str(file_path),
                'filename': Path(file_path).name,
                'title': Path(file_path).stem,
                'description': f"Text template: {Path(file_path).name}",
                'type': 'text_template'
            }
        else:
            print(f"❌ MJML TEXT templates NOT found")
            # Original HTML processing
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
            
            # Create structured content with better formatting
            structured_content = f"""
=== TEMPLATE METADATA ===
File: {Path(file_path).name}
Title: {title_text}
Description: {description}
Classes: {', '.join(list(classes)[:10]) if classes else 'None'}
IDs: {', '.join(ids) if ids else 'None'}
Headers: {'; '.join(sections) if sections else 'None'}
Content Length: {len(content)} characters

=== HTML STRUCTURE ===
{content}

=== END OF TEMPLATE ===
            """.strip()
            
            return structured_content, {
                'source': str(file_path),
                'filename': Path(file_path).name,
                'title': title_text,
                'description': description,
                'type': 'html_template'
            }

    def ingest_templates(self, templates_dir="./dynamictemplates"):
        """Ingest all HTML templates from directory"""
        import shutil
        import time

        # Use the same method as create_sample_templates_from_folder for consistency
        if not os.path.exists(templates_dir):
            print(f"❌ Directory '{templates_dir}' does not exist!")
            return None
        
        # Find all HTML/MJML/TXT files in the directory
        all_files = []
        for filename in os.listdir(templates_dir):
            file_path = os.path.join(templates_dir, filename)
            if filename.endswith((".html", ".htm", ".mjml", ".txt")) and os.path.isfile(file_path):
                all_files.append(file_path)

        # Also try glob as backup (for subdirectories)
        patterns = ["*.html", "*.htm", "*.mjml", "*.txt"]
        for pattern in patterns:
            glob_files = glob.glob(os.path.join(templates_dir, "**", pattern), recursive=True)
            for f in glob_files:
                if f not in all_files:
                    all_files.append(f)

        if not all_files:
            print(f"❌ No matching templates found in '{templates_dir}'")
            return None

        print(f"Found {len(all_files)} matching templates:")
        for f in all_files:
            print(f" - {f} ({os.path.getsize(f):,} bytes)")

        # Force-clear Chroma DB, even if locked
        if os.path.exists(self.persist_directory):
            for attempt in range(3):
                try:
                    print(f"Clearing previous Chroma DB at {self.persist_directory}...")
                    shutil.rmtree(self.persist_directory)
                    break
                except PermissionError:
                    print(f"⚠ DB folder in use. Retrying in 1s... (Attempt {attempt+1}/3)")
                    time.sleep(1)
            else:
                print("❌ Could not delete old Chroma DB. Please close any processes using it.")
                return None

        documents = []
        processed_files = []

        for file_path in all_files:
            print(f"Processing: {file_path}")
            try:
                content, metadata = self.extract_html_info(file_path)
                processed_files.append(file_path)

                texts = self.text_splitter.split_text(content)
                print(f"  → Split into {len(texts)} chunks")

                for i, text in enumerate(texts):
                    doc_metadata = metadata.copy()
                    doc_metadata['chunk_id'] = i
                    doc_metadata['total_chunks'] = len(texts)
                    doc_metadata['chunk_size'] = len(text)
                    doc_metadata['chunk_start'] = i * (int(os.getenv("CHUNK_SIZE", 2000)) - int(os.getenv("CHUNK_OVERLAP", 300)))
                    # Add file index for uniqueness
                    doc_metadata['file_index'] = len(processed_files) - 1

                    documents.append(Document(
                        page_content=text,
                        metadata=doc_metadata
                    ))

            except Exception as e:
                print(f"  ❌ Error processing {file_path}: {e}")

        print(f"\n✅ Successfully processed {len(processed_files)} files:")
        for i, fp in enumerate(processed_files):
            print(f"  {i+1}. {Path(fp).name}")

        if not documents:
            print("❌ No documents prepared for ingestion.")
            return None

        # Show unique sources before saving
        unique_sources = set(doc.metadata['source'] for doc in documents)
        print(f"\nUnique sources in this ingestion ({len(unique_sources)} files):")
        for src in unique_sources:
            print(" -", src)

        print(f"\nCreating vector store with {len(documents)} document chunks...")
        vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            persist_directory=self.persist_directory
        )

        print(f"✅ Successfully ingested {len(documents)} chunks from templates")
        return vectorstore


def inspect_vectorstore(persist_directory, limit=10):
    print("\n" + "=" * 50)
    print("INSPECTING STORED CHROMA CONTENT")
    print("=" * 50)

    # Reopen the DB without embeddings (we only want to read metadata & docs)
    db = Chroma(persist_directory=persist_directory, embedding_function=None)

    try:
        results = db.get()
        total_docs = len(results['documents'])
        print(f"Total stored documents: {total_docs}\n")

        # Count documents per file
        file_counts = {}
        for metadata in results['metadatas']:
            filename = metadata.get('filename', 'Unknown')
            file_counts[filename] = file_counts.get(filename, 0) + 1
        
        print("📊 DOCUMENT DISTRIBUTION BY FILE:")
        for filename, count in sorted(file_counts.items()):
            print(f"  {filename}: {count} chunks")
        print()

        # Show sample documents
        print("📋 SAMPLE DOCUMENTS:")
        for i in range(min(limit, total_docs)):
            print(f"--- Document #{i+1} ---")
            print(f"ID: {results['ids'][i]}")
            print(f"Metadata: {results['metadatas'][i]}")
            print(f"Preview: {results['documents'][i][:300]}...")
            print()
    except Exception as e:
        print(f"Error reading ChromaDB: {e}")

def main():
    """Main function to run the ingestion process"""
    templates_dir = "dynamictemplates"
    
    # Initialize the ingester
    ingester = HTMLTemplateIngester()
    
    # Add debug option
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--debug":
        print("=== DEBUG MODE ===")
        filename = sys.argv[2] if len(sys.argv) > 2 else None
        ingester.debug_template_content(templates_dir, filename)
        return
    
    # Load templates from folder first (for verification)
    print("=" * 50)
    print("LOADING TEMPLATES FROM FOLDER")
    print("=" * 50)
    templates = ingester.create_sample_templates_from_folder(templates_dir)
    
    if not templates:
        print(f"❌ No templates found in '{templates_dir}' directory.")
        print("Please add template files (.html, .htm, .mjml, .txt) to the directory and run again.")
        return None
    
    # Display loaded templates with size info
    total_size = 0
    for filename, content in templates.items():
        size = len(content)
        total_size += size
        file_type = "TEXT" if filename.endswith('.txt') else "HTML"
        print(f"✓ {filename} ({size:,} characters) [{file_type}]")
    
    print(f"\nTotal content: {total_size:,} characters across {len(templates)} templates")
    
    print("\n" + "=" * 50)
    print("INGESTING TEMPLATES INTO VECTOR STORE")
    print("=" * 50)
    
    # Ingest templates into vector store
    vectorstore = ingester.ingest_templates(templates_dir)
    
    if vectorstore:
        print("\n" + "=" * 50)
        print("INGESTION COMPLETE")
        print("=" * 50)
        print(f"✓ Vector store saved to: {ingester.persist_directory}")
        print("✓ Ready for RAG queries!")
        inspect_vectorstore(ingester.persist_directory, limit=10)
        
        # Show chunking statistics
        print(f"\nChunking Settings:")
        print(f"  - Chunk size: {os.getenv('CHUNK_SIZE', 2000)} characters")
        print(f"  - Chunk overlap: {os.getenv('CHUNK_OVERLAP', 300)} characters")
    
    return vectorstore

if __name__ == "__main__":
    vectorstore = main()