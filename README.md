---
title: HTML Email RAG System
emoji: 🌐
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 4.44.0
app_file: streamlit_demo.py
pinned: false
python_version: 3.10
---

# HTML Email RAG System 🌐

An intelligent HTML template retrieval and generation system powered by RAG (Retrieval-Augmented Generation) that helps you find, understand, and create HTML templates and email layouts.

## 🚀 Features

- **Smart Template Search**: Query existing HTML templates using natural language
- **HTML Generation**: Generate new HTML code based on your requirements
- **Email Template Support**: Specialized support for email templates and layouts
- **RAG-Powered**: Uses Retrieval-Augmented Generation for accurate, context-aware responses
- **Multiple Models**: Supports both Ollama (tinyllama) and HuggingFace models
- **Vector Database**: ChromaDB for efficient template storage and retrieval
- **Web Interface**: Streamlit-based demo interface for easy interaction

## 🏗️ Architecture

This system consists of:

- **FastAPI Backend** (`main_api.py`): RESTful API for template queries and HTML generation
- **Streamlit Frontend** (`streamlit_demo.py`): User-friendly web interface
- **Ingestion System** (`ingest_script.py`): Processes and indexes HTML templates
- **Vector Database**: ChromaDB with HuggingFace embeddings for semantic search
- **LLM Integration**: Supports Ollama and HuggingFace models for text generation

## 📋 Prerequisites

- Python 3.8+
- Hugging Face account
- Internet connection for model downloads

## 🛠️ Installation & Setup

### 1. Clone the Repository
```bash
git clone https://huggingface.co/spaces/kpomservices/HTML-EMAIL-RAG
cd HTML-EMAIL-RAG
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Prepare Templates
The system comes with sample templates in the `templates/` directory:
- `dashboard.html` - Dashboard layout template
- `email_template.html` - Email template
- `landing_page.html` - Landing page template

### 4. Ingest Templates
```bash
python ingest_script.py
```

This will:
- Process all HTML files in the `templates/` directory
- Extract metadata, classes, IDs, and content
- Create embeddings using HuggingFace's `all-MiniLM-L6-v2` model
- Store everything in a ChromaDB vector database

### 5. Start the API Server
```bash
python main_api.py
```

The API will be available at `http://localhost:8000`

### 6. Launch the Web Interface
```bash
streamlit run streamlit_demo.py
```

## 🎯 Usage

### Web Interface

1. **Query Templates**: Ask questions about existing templates
   - "Show me a navigation bar structure"
   - "How to create a responsive grid layout?"
   - "What CSS classes are used for buttons?"

2. **Generate HTML**: Create new HTML code
   - "Create a responsive email template"
   - "Generate a dashboard with sidebar"
   - "Make a landing page with hero section"

### API Endpoints

#### Health Check
```bash
GET /health
```

#### Query Templates
```bash
POST /query
{
  "query": "How to create a responsive navigation bar?",
  "max_results": 3,
  "include_sources": true
}
```

#### Generate HTML
```bash
POST /generate
{
  "query": "Create a modern email template with header and footer",
  "max_results": 5,
  "include_sources": true
}
```

## 🔧 Configuration

### Environment Variables
Create a `.env` file with:
```env
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
EMBEDDING_MODEL=all-MiniLM-L6-v2
LLM_MODEL=tinyllama
```

### Model Options

The system supports multiple LLM backends:

1. **Ollama (Recommended)**: Fast and reliable
   - Model: `tinyllama`
   - Context window: 4096 tokens
   - Max tokens: 2048

2. **HuggingFace Fallback**: When Ollama is not available
   - Model: `distilgpt2`
   - Max tokens: 1024
   - Temperature: 0.1

## 📁 Project Structure

```
HTML-EMAIL-RAG/
├── main_api.py              # FastAPI backend
├── streamlit_demo.py        # Streamlit frontend
├── ingest_script.py         # Template ingestion
├── requirements.txt         # Python dependencies
├── templates/               # HTML template files
│   ├── dashboard.html
│   ├── email_template.html
│   └── landing_page.html
├── chroma_db/              # Vector database (created after ingestion)
└── README.md               # This file
```

## 🎨 Template Features

The system can handle various HTML template types:

- **Email Templates**: Responsive email layouts with inline CSS
- **Dashboard Templates**: Admin panels and data visualization
- **Landing Pages**: Marketing and conversion-focused pages
- **Component Templates**: Reusable UI components

## 🔍 Search Capabilities

The RAG system can answer questions about:

- HTML structure and semantics
- CSS classes and styling patterns
- JavaScript functionality
- Responsive design techniques
- Accessibility features
- Email compatibility
- Cross-browser support

## 🚀 Deployment on Hugging Face Spaces

This application is configured to run on Hugging Face Spaces with:

- **Runtime**: Python 3.10
- **SDK**: Gradio (for the web interface)
- **Hardware**: CPU (free tier) or GPU (if needed)
- **Auto-deploy**: Changes to main branch trigger automatic deployment

## 🤝 Contributing

1. Fork the repository
2. Add your HTML templates to the `templates/` directory
3. Run the ingestion script to update the vector database
4. Test your changes
5. Submit a pull request

## 📝 License

This project is open source and available under the MIT License.

## 🆘 Support

If you encounter any issues:

1. Check the API health endpoint: `/health`
2. Ensure templates are properly ingested
3. Verify all dependencies are installed
4. Check the logs for error messages

## 🔗 Links

- **Hugging Face Space**: https://huggingface.co/spaces/kpomservices/HTML-EMAIL-RAG
- **API Documentation**: Available at `/docs` when the server is running
- **Health Check**: Available at `/health`

---

**Happy HTML templating! 🎉** 