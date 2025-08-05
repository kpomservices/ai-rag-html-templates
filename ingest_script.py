import os
import glob
from pathlib import Path
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

class HTMLTemplateIngester:
    def __init__(self, persist_directory="./chroma_db"):
        self.persist_directory = persist_directory

        self.embeddings = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')

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
    
    def create_sample_templates(self, templates_dir):
        """Create sample HTML templates for demo"""
        os.makedirs(templates_dir, exist_ok=True)
        
        samples = {
            "landing_page.html": """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="description" content="Modern landing page template with hero section">
    <title>Landing Page Template</title>
    <style>
        .hero { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); }
        .cta-button { background-color: #ff6b6b; color: white; padding: 12px 24px; }
        .features-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 2rem; }
    </style>
</head>
<body>
    <header class="navigation">
        <nav class="navbar">
            <div class="logo">Brand</div>
            <ul class="nav-links">
                <li><a href="#home">Home</a></li>
                <li><a href="#features">Features</a></li>
                <li><a href="#contact">Contact</a></li>
            </ul>
        </nav>
    </header>
    
    <section class="hero">
        <div class="hero-content">
            <h1>Welcome to Our Platform</h1>
            <p>Build amazing things with our modern solution</p>
            <button class="cta-button">Get Started</button>
        </div>
    </section>
    
    <section class="features">
        <div class="features-grid">
            <div class="feature-card">
                <h3>Fast Performance</h3>
                <p>Lightning-fast loading times</p>
            </div>
            <div class="feature-card">
                <h3>Responsive Design</h3>
                <p>Works on all devices</p>
            </div>
            <div class="feature-card">
                <h3>Easy to Use</h3>
                <p>Intuitive user interface</p>
            </div>
        </div>
    </section>
</body>
</html>""",
            
            "email_template.html": """<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="description" content="Responsive email template for newsletters">
    <title>Email Newsletter Template</title>
    <style>
        .email-container { max-width: 600px; margin: 0 auto; font-family: Arial, sans-serif; }
        .header { background-color: #2c3e50; color: white; padding: 20px; text-align: center; }
        .content { padding: 20px; background-color: #f8f9fa; }
        .footer { background-color: #34495e; color: white; padding: 15px; text-align: center; }
        .button { background-color: #3498db; color: white; padding: 10px 20px; text-decoration: none; border-radius: 4px; }
    </style>
</head>
<body>
    <div class="email-container">
        <div class="header">
            <h1>Company Newsletter</h1>
        </div>
        
        <div class="content">
            <h2>Latest Updates</h2>
            <p>Here are the latest news and updates from our team.</p>
            
            <div class="news-item">
                <h3>New Feature Launch</h3>
                <p>We're excited to announce our new dashboard feature.</p>
                <a href="#" class="button">Learn More</a>
            </div>
            
            <div class="news-item">
                <h3>Upcoming Webinar</h3>
                <p>Join us for an exclusive webinar next week.</p>
                <a href="#" class="button">Register Now</a>
            </div>
        </div>
        
        <div class="footer">
            <p>&copy; 2024 Company Name. All rights reserved.</p>
            <p><a href="#" style="color: #bdc3c7;">Unsubscribe</a></p>
        </div>
    </div>
</body>
</html>""",
            
            "dashboard.html": """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="description" content="Admin dashboard template with sidebar and charts">
    <title>Dashboard Template</title>
    <style>
        .dashboard { display: flex; min-height: 100vh; }
        .sidebar { width: 250px; background-color: #2c3e50; color: white; padding: 20px; }
        .main-content { flex: 1; padding: 20px; background-color: #ecf0f1; }
        .stats-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 20px; margin-bottom: 30px; }
        .stat-card { background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .chart-container { background: white; padding: 20px; border-radius: 8px; margin-bottom: 20px; }
        .nav-item { padding: 10px 0; border-bottom: 1px solid #34495e; }
    </style>
</head>
<body>
    <div class="dashboard">
        <aside class="sidebar">
            <h2>Admin Panel</h2>
            <nav>
                <div class="nav-item"><a href="#dashboard">Dashboard</a></div>
                <div class="nav-item"><a href="#users">Users</a></div>
                <div class="nav-item"><a href="#analytics">Analytics</a></div>
                <div class="nav-item"><a href="#settings">Settings</a></div>
            </nav>
        </aside>
        
        <main class="main-content">
            <h1>Dashboard Overview</h1>
            
            <div class="stats-grid">
                <div class="stat-card">
                    <h3>Total Users</h3>
                    <p class="stat-number">1,234</p>
                </div>
                <div class="stat-card">
                    <h3>Revenue</h3>
                    <p class="stat-number">$12,345</p>
                </div>
                <div class="stat-card">
                    <h3>Orders</h3>
                    <p class="stat-number">567</p>
                </div>
                <div class="stat-card">
                    <h3>Growth</h3>
                    <p class="stat-number">+23%</p>
                </div>
            </div>
            
            <div class="chart-container">
                <h3>Sales Analytics</h3>
                <div id="chart-placeholder" style="height: 300px; background: #f8f9fa; display: flex; align-items: center; justify-content: center;">
                    Chart placeholder - integrate with your preferred charting library
                </div>
            </div>
        </main>
    </div>
</body>
</html>"""
        }
        
        for filename, content in samples.items():
            with open(os.path.join(templates_dir, filename), 'w', encoding='utf-8') as f:
                f.write(content)
        
        print(f"Created {len(samples)} sample templates in {templates_dir}")

if __name__ == "__main__":
    ingester = HTMLTemplateIngester()
    vectorstore = ingester.ingest_templates()
