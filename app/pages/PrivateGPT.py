import os
from typing import Final

import streamlit as st
from langchain.callbacks.base import BaseCallbackHandler
from langchain.chat_models.ollama import ChatOllama
from langchain.embeddings import CacheBackedEmbeddings
from langchain.embeddings.ollama import OllamaEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.schema import Document
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.storage import LocalFileStore
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders.unstructured import UnstructuredFileLoader
from langchain_community.vectorstores.faiss import FAISS

FILES_DIR: Final[str] = ".cache/private_files"
EMBEDDINGS_DIR: Final[str] = ".cache/private_embeddings"

st.set_page_config(
    page_title="PrivateGPT",
    page_icon="📄",
    layout="centered",
    initial_sidebar_state="auto",
)
st.title("PrivateGPT")
st.markdown(
    """
Welcome!

Use this chatbot to ask questions to an AI about your files! 🤖
"""
)


class ChatCallbackHandler(BaseCallbackHandler):
    message = ""

    def on_llm_start(self, *args, **kwargs):
        self.message_box = st.empty()

    def on_llm_end(self, *args, **kwargs):
        save_message(self.message, "ai")

    def on_llm_new_token(self, token: str, *args, **kwargs):
        self.message += token
        self.message_box.markdown(self.message)


llm = ChatOllama(
    model="mistral:latest",
    temperature=0.1,
    callbacks=[
        ChatCallbackHandler(),
    ],
)

with st.sidebar:
    file = st.file_uploader("Upload a .txt file", type=["txt"])


@st.cache_resource(show_spinner="Embedding file...")
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
    embeddings = OllamaEmbeddings(model="mistral:latest")
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_path)
    vector_store = FAISS.from_documents(docs, cached_embeddings)
    retriever = vector_store.as_retriever()
    return retriever


session_messages = st.session_state.get("messages", [])


def save_message(message, role: str):
    session_messages.append({"role": role, "message": message})


def send_message(message, role: str, save: bool = True):
    with st.chat_message(role):
        st.markdown(message)
    if save:
        save_message(message, role)


def paint_history():
    for message in session_messages:
        send_message(message["message"], message["role"], save=False)


def format_documents(documents: list[Document]):
    return "\n\n".join([doc.page_content for doc in documents])


prompt = ChatPromptTemplate.from_template(
    """Answer the question using ONLY the following context and not your training data. If you don't know the answer, just say "I don't know". DON'T make anything up.

Context:
{context}

Question:
{question}
"""
)


if file:
    retriever = embed_file(file)
    send_message("I'm ready! Ask me anything about the file.", "ai", save=False)
    paint_history()
    message = st.chat_input("Send message to the AI")
    if message:
        send_message(message, "human")
        chain = (
            {
                "context": retriever | RunnableLambda(format_documents),
                "question": RunnablePassthrough(),
            }
            | prompt
            | llm
        )
        with st.chat_message("ai"):
            response = chain.invoke(message)

else:
    st.session_state["messages"] = []
