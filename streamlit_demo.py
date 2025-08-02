import streamlit as st
import requests
import json
from dotenv import load_dotenv

load_dotenv()

# Configure Streamlit page
st.set_page_config(
    page_title="HTML Template RAG Demo",
    page_icon="🌐",
    layout="wide"
)

# API base URL
API_BASE_URL = "http://localhost:8000"

def check_api_health():
    """Check if the API is running"""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.status_code == 200, response.json()
    except requests.exceptions.RequestException:
        return False, {"error": "API not reachable"}

def query_api(endpoint, query, max_results=3, include_sources=True):
    """Make API request"""
    try:
        payload = {
            "query": query,
            "max_results": max_results,
            "include_sources": include_sources
        }
        response = requests.post(
            f"{API_BASE_URL}/{endpoint}",
            json=payload,
            timeout=3600  # Increased timeout to 2 minutes
        )
        response.raise_for_status()
        return True, response.json()
    except requests.exceptions.RequestException as e:
        return False, {"error": str(e)}

# Main UI
st.title("🌐 HTML Template RAG System")
st.markdown("Query HTML templates and generate new HTML code using AI")

# Check API status
with st.sidebar:
    st.header("System Status")
    api_healthy, health_data = check_api_health()
    
    if api_healthy:
        st.success("✅ API is running")
        if health_data.get("documents_available"):
            st.success("✅ Templates loaded")
        else:
            st.warning("⚠️ No templates found")
            st.info("Run `python ingest.py` first")
    else:
        st.error("❌ API not available")
        st.info("Run `python main.py` to start the API")
    
    st.markdown("---")
    st.header("Settings")
    max_results = st.slider("Max results", 1, 10, 3)
    include_sources = st.checkbox("Include sources", value=True)

# Main content area
tab1, tab2 = st.tabs(["Query Templates", "Generate HTML"])

with tab1:
    st.header("Ask Questions About Templates")
    st.markdown("Ask questions about existing HTML templates, styling, structure, etc.")
    
    # Example queries
    with st.expander("Example Queries"):
        st.markdown("""
        - "Show me a navigation bar structure"
        - "How to create a responsive grid layout?"
        - "What CSS classes are used for buttons?"
        - "Show me email template structure"
        - "How to create a dashboard sidebar?"
        """)
    
    query_input = st.text_area(
        "Your question:",
        placeholder="e.g., How do I create a responsive navigation bar?",
        height=100
    )
    
    if st.button("Search Templates", type="primary"):
        if not query_input.strip():
            st.warning("Please enter a question")
        elif not api_healthy:
            st.error("API is not available. Please start the API server.")
        else:
            with st.spinner("Searching templates..."):
                success, result = query_api("query", query_input, max_results, include_sources)
                
                if success:
                    st.success("Found relevant information!")
                    
                    # Display answer
                    st.subheader("Answer")
                    st.markdown(result["answer"])
                    
                    # Display sources
                    if include_sources and result.get("sources"):
                        st.subheader("Sources")
                        for i, source in enumerate(result["sources"], 1):
                            with st.expander(f"Source {i}: {source['filename']}"):
                                st.markdown(f"**Title:** {source.get('title', 'N/A')}")
                                st.markdown(f"**File:** {source['filename']}")
                                st.markdown("**Preview:**")
                                st.code(source["content_preview"], language="html")
                else:
                    error_msg = result.get('error', 'Unknown error')
                    st.error(f"Error: {error_msg}")
                    
                    # Provide helpful error messages
                    if "timeout" in error_msg.lower():
                        st.info("💡 **Timeout Tips:**\n- Try a shorter query\n- Check if Ollama is running\n- Restart the API server")
                    elif "ollama" in error_msg.lower():
                        st.info("💡 **Ollama Issue:**\n- Install Ollama from https://ollama.ai\n- Run `ollama pull tinyllama`\n- Make sure Ollama service is running")

with tab2:
    st.header("Generate HTML Code")
    st.markdown("Generate new HTML code based on your requirements")
    
    # Example prompts
    with st.expander("Example Prompts"):
        st.markdown("""
        - "Create a landing page with hero section and features"
        - "Build a contact form with validation styling"
        - "Generate a pricing table with 3 columns"
        - "Create a blog post layout with sidebar"
        - "Build a responsive photo gallery"
        """)
    
    generate_input = st.text_area(
        "Describe what you want to create:",
        placeholder="e.g., Create a modern pricing table with 3 tiers",
        height=100
    )
    
    if st.button("Generate HTML", type="primary"):
        if not generate_input.strip():
            st.warning("Please describe what you want to create")
        elif not api_healthy:
            st.error("API is not available. Please start the API server.")
        else:
            with st.spinner("Generating HTML code..."):
                success, result = query_api("generate", generate_input, max_results, include_sources)
                
                if success:
                    st.success("HTML generated successfully!")
                    
                    # Display generated HTML
                    if result.get("generated_html"):
                        st.subheader("Generated HTML")
                        
                        # Show code
                        st.code(result["generated_html"], language="html")
                        
                        # Show preview
                        st.subheader("Preview")
                        st.components.v1.html(result["generated_html"], height=400, scrolling=True)
                        
                        # Download button
                        st.download_button(
                            label="Download HTML",
                            data=result["generated_html"],
                            file_name="generated.html",
                            mime="text/html"
                        )
                    
                    # Show context answer
                    if result.get("answer"):
                        st.subheader("Context & Explanation")
                        st.markdown(result["answer"])
                    
                    # Show sources
                    if include_sources and result.get("sources"):
                        with st.expander("Reference Templates"):
                            for i, source in enumerate(result["sources"], 1):
                                st.markdown(f"**{i}. {source['filename']}**")
                                st.markdown(f"Title: {source.get('title', 'N/A')}")
                                st.code(source["content_preview"], language="html")
                                st.markdown("---")
                else:
                    st.error(f"Error: {result.get('error', 'Unknown error')}")

# Footer
st.markdown("---")
st.markdown("""
### Quick Start:
1. Install dependencies: `pip install -r requirements.txt`
2. Set up your OpenAI API key in `.env` file
3. Run ingestion: `python ingest.py`
4. Start API: `python main.py`
5. Launch demo: `streamlit run demo.py`
""")

# Status indicator
if api_healthy:
    st.sidebar.success("🟢 System Ready")
else:
    st.sidebar.error("🔴 System Not Ready")