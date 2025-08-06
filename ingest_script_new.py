from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
import os

class MJMLQueryHandler:
    def __init__(self, persist_directory="./chroma_db"):
        self.persist_directory = persist_directory
        self.embeddings = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')
        
        # Load existing vector store
        try:
            self.vectorstore = Chroma(
                persist_directory=persist_directory,
                embedding_function=self.embeddings
            )
            print(f"✓ Loaded vector store from {persist_directory}")
        except Exception as e:
            print(f"❌ Error loading vector store: {e}")
            print("Please run the ingestion script first.")
            self.vectorstore = None
    
    def enhance_query_for_mjml(self, query):
        """Enhance user query to prioritize MJML results"""
        mjml_keywords = ["mjml", "email template", "responsive email", "mj-"]
        
        # Check if query already mentions MJML
        if any(keyword in query.lower() for keyword in mjml_keywords):
            return query
        
        # Enhance query with MJML context
        enhanced_query = f"MJML email template {query} responsive email"
        return enhanced_query
    
    def retrieve_mjml_templates(self, query, k=5):
        """Retrieve relevant MJML templates with filtering"""
        if not self.vectorstore:
            return []
        
        # Enhance query for better MJML matching
        enhanced_query = self.enhance_query_for_mjml(query)
        
        # Retrieve documents
        docs = self.vectorstore.similarity_search(enhanced_query, k=k*2)  # Get more to filter
        
        # Filter and prioritize MJML templates
        mjml_docs = []
        html_docs = []
        
        for doc in docs:
            if doc.metadata.get('is_mjml', False) or doc.metadata.get('format') == 'MJML':
                mjml_docs.append(doc)
            else:
                html_docs.append(doc)
        
        # Prioritize MJML documents
        final_docs = mjml_docs[:k] if mjml_docs else html_docs[:k]
        
        print(f"Retrieved {len(mjml_docs)} MJML docs and {len(html_docs)} HTML docs")
        print(f"Using top {len(final_docs)} documents for generation")
        
        return final_docs
    
    def create_mjml_prompt(self, query, retrieved_docs):
        """Create a comprehensive prompt for MJML generation"""
        
        # Collect relevant template content
        template_examples = []
        for doc in retrieved_docs:
            metadata = doc.metadata
            content_preview = doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content
            
            template_examples.append(f"""
Template: {metadata.get('filename', 'Unknown')}
Format: {metadata.get('format', 'Unknown')}
Components: {', '.join(metadata.get('mjml_components', []))}
Keywords: {', '.join(metadata.get('semantic_keywords', []))}

Content Preview:
{content_preview}
""")
        
        examples_text = "\n".join(template_examples)
        
        prompt_template = f"""
You are an expert MJML email template developer. Based on the following MJML template examples and the user's request, generate a complete, working MJML template.

USER REQUEST: {query}

RELEVANT MJML TEMPLATES:
{examples_text}

REQUIREMENTS:
1. Generate ONLY MJML code (not HTML)
2. Start with proper MJML doctype: <!doctype html>
3. Use proper MJML structure: <mjml><mj-head>...<mj-body>...
4. Use MJML components: mj-section, mj-column, mj-text, mj-button, mj-table, etc.
5. Make it responsive and email-client friendly
6. Include proper styling within MJML attributes
7. For tables/pricing, use mj-table or structured mj-section/mj-column layout
8. Include proper spacing and typography

SPECIFIC FOCUS for the request "{query}":
- If creating a pricing table, use multiple mj-columns within mj-sections
- Use mj-table for tabular data if needed
- Include mj-button for call-to-action elements
- Ensure responsive design with proper column layouts
- Use appropriate MJML styling attributes

Generate a complete, production-ready MJML template:
"""
        
        return prompt_template
    
    def query_mjml_template(self, user_query):
        """Main method to query and generate MJML templates"""
        print(f"Processing query: {user_query}")
        
        # Retrieve relevant documents
        docs = self.retrieve_mjml_templates(user_query)
        
        if not docs:
            return "No relevant MJML templates found. Please ensure MJML templates are ingested."
        
        # Create prompt
        prompt = self.create_mjml_prompt(user_query, docs)
        
        return {
            'prompt': prompt,
            'retrieved_docs': docs,
            'enhanced_query': self.enhance_query_for_mjml(user_query)
        }
    
    def get_template_stats(self):
        """Get statistics about ingested templates"""
        if not self.vectorstore:
            return "Vector store not loaded"
        
        # Get all documents
        all_docs = self.vectorstore.similarity_search("", k=1000)
        
        mjml_count = len([d for d in all_docs if d.metadata.get('is_mjml', False)])
        html_count = len(all_docs) - mjml_count
        
        unique_files = set(d.metadata.get('filename', 'Unknown') for d in all_docs)
        
        return f"""
Template Statistics:
- Total chunks: {len(all_docs)}
- MJML chunks: {mjml_count}
- HTML chunks: {html_count}
- Unique files: {len(unique_files)}
- Files: {', '.join(unique_files)}
        """

# Example usage
def main():
    handler = MJMLQueryHandler()
    
    # Show stats
    print(handler.get_template_stats())
    
    # Example queries
    test_queries = [
        "Generate a pricing table with 3 columns",
        "Create MJML pricing table with 3 columns", 
        "MJML email template responsive pricing comparison"
    ]
    
    for query in test_queries:
        print(f"\n{'='*50}")
        print(f"Testing query: {query}")
        print('='*50)
        
        result = handler.query_mjml_template(query)
        if isinstance(result, dict):
            print("Enhanced query:", result['enhanced_query'])
            print(f"Retrieved {len(result['retrieved_docs'])} documents")
            print("\nGenerated prompt preview:")
            print(result['prompt'][:500] + "...")
        else:
            print(result)

if __name__ == "__main__":
    main()