import streamlit as st
import wikipedia
from langchain.document_loaders.unstructured import UnstructuredFileLoader
from langchain.retrievers.wikipedia import WikipediaRetriever
from langchain_text_splitters import CharacterTextSplitter
from streamlit.runtime.uploaded_file_manager import UploadedFile

st.title("DocumentGPT")


@st.cache_resource(show_spinner="Loading file...")
def split_file(file: UploadedFile):
    file_content = file.read()
    file_path = f"./.cache/quiz_files/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size=600,
        chunk_overlap=100,
    )
    loader = UnstructuredFileLoader(file_path)
    docs = loader.load_and_split(text_splitter=splitter)
    return docs


with st.sidebar:
    choice = st.selectbox(
        "Choose what you want to use.",
        (
            "File",
            "Wikipedia Article",
        ),
    )

    if choice == "File":
        file = st.file_uploader("Upload a file", type=["txt"])
        if file:
            docs = split_file(file)

    else:
        topic = st.text_input("Search wikipedia")
        if topic:
            retriever = WikipediaRetriever(wiki_client=wikipedia, top_k_results=5)
            with st.status("Searching Wikipedia..."):
                retriever.get_relevant_documents(query=topic)
