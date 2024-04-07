import streamlit as st
import wikipedia
from langchain.retrievers.wikipedia import WikipediaRetriever


@st.cache_data(show_spinner="Searching Wikipedia...")
def wiki_search(term: str):
    retriever = WikipediaRetriever(wiki_client=wikipedia, top_k_results=5)
    docs = retriever.get_relevant_documents(term)
    return docs
