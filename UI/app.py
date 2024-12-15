import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import streamlit as st
import logging
import os
from dotenv import load_dotenv
from RAG.rag_agent import RAGAgent

load_dotenv()

rag_agent = RAGAgent()

def get_rag_response(query: str) -> str:
    """
    Get a response from the RAG pipeline for a given query.
    """
    response = rag_agent.answer(query=query, top_k=5)
    return response if response else "I'm sorry, I don't have information on that topic at the moment."

# Set page configuration
st.set_page_config(
    page_title="RAG Application",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for enhanced styling
def local_css(file_name):
    if os.path.exists(file_name):
        with open(file_name) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    else:
        st.warning(f"CSS file {file_name} not found. Please ensure it exists.")

# Apply custom CSS
local_css("styles.css")

# Initialize logging
logger = logging.getLogger("app")
logger.setLevel(logging.DEBUG)

# Initialize session state
if "initialized" not in st.session_state or not st.session_state.initialized:
    logger.debug("Initializing session state")
    st.session_state.messages = []
    st.session_state.initialized = True

# Header Section
def header_section():
    st.markdown("<h1 class='main-header'>Financial RAG Application</h1>", unsafe_allow_html=True)
    st.markdown("<h3 class='sub-header'>Retrieve and Generate Insights from Your Financial Documents</h3>", unsafe_allow_html=True)

# Input Query Section
def query_input_section():
    st.markdown("<hr class='divider'>", unsafe_allow_html=True)
    st.markdown("<h2 class='search-title'>Enter Your Query</h2>", unsafe_allow_html=True)
    query = st.text_area(
        "Ask a question:",
        height=200,
        key="query",
        label_visibility="collapsed",
        placeholder="Enter your query here..."
    )
    return query

# Search Button
def search_button(query):
    if st.button("Search", key="search_button"):
        if not query.strip():
            st.warning("Please enter a valid query.")
        else:
            with st.spinner("Processing your query..."):
                response = get_rag_response(query)
                if response:
                    st.success("Here is your answer:")
                    st.markdown(f"<div class='result-text'>{response}</div>", unsafe_allow_html=True)
                else:
                    st.error("No response found. Please try a different query.")

# Footer Section
def footer_section():
    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

# Main Function to Organize Sections
def main():
    header_section()
    query = query_input_section()
    search_button(query)
    footer_section()

if __name__ == "__main__":
    main()
