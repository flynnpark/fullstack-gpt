import os
import time
from typing import Final

import streamlit as st
from langchain.embeddings import CacheBackedEmbeddings
from langchain.storage import LocalFileStore
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders.unstructured import UnstructuredFileLoader
from langchain_community.vectorstores.faiss import FAISS
from langchain_openai import OpenAIEmbeddings

FILES_DIR: Final[str] = ".cache/files"
EMBEDDINGS_DIR: Final[str] = ".cache/embeddings"

st.set_page_config(
    page_title="DocumentGPT",
    page_icon="📄",
    layout="centered",
    initial_sidebar_state="auto",
)
st.title("DocumentGPT")
st.markdown(
    """
Welcome!

Use this chatbot to ask questions to an AI about your files! 🤖
"""
)

file = st.file_uploader("Upload a .txt file", type=["txt"])


def embed_file(file):
    file_content = file.read()
    file_path = f"{FILES_DIR}/{file.name}"
    if not os.path.exists(FILES_DIR):
        os.makedirs(FILES_DIR)
    with open(file_path, "wb") as f:
        f.write(file_content)

    cache_path = LocalFileStore(f"{EMBEDDINGS_DIR}/{file.name}")
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size=600,
        chunk_overlap=100,
    )
    loader = UnstructuredFileLoader(file_path)
    docs = loader.load_and_split(splitter)
    embeddings = OpenAIEmbeddings()
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_path)
    vector_store = FAISS.from_documents(docs, cached_embeddings)
    retriever = vector_store.as_retriever()
    return retriever


if file:
    retriever = embed_file(file)


if "messages" not in st.session_state:
    st.session_state["messages"] = []

session_messages = st.session_state["messages"]


def send_message(message: str, role: str, save: bool = True):
    with st.chat_message(role):
        st.write(message)
        if save:
            session_messages.append({"role": role, "message": message})


for message in session_messages:
    send_message(message["message"], message["role"], save=False)

message = st.chat_input("Send message to the AI")
if message:
    send_message(message, "human")
    time.sleep(2)
    send_message("I am a robot 🤖", "ai")
