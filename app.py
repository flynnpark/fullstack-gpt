import os

import streamlit as st
from langchain.document_loaders.unstructured import UnstructuredFileLoader
from langchain.embeddings import CacheBackedEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.storage import LocalFileStore
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.chroma import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from streamlit.runtime.uploaded_file_manager import UploadedFile

st.set_page_config(
    page_title="Streamlit App",
    page_icon="🧊",
)


llm = ChatOpenAI(
    temperature=0.1,
    api_key=st.session_state.get("openai_key"),
)


def make_dir(file_path: str):
    if not os.path.exists(file_path):
        os.makedirs(os.path.dirname(file_path))


@st.cache_resource(show_spinner="Embedding file...")
def embed_file(file: UploadedFile):
    file_content = file.read()
    file_path = f".cache/files/{file.name}"
    make_dir(file_path)
    with open(file_path, "wb") as f:
        f.write(file_content)
    cache_dir = LocalFileStore(f"./.cache/embeddings/{file.name}")
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size=600,
        chunk_overlap=100,
    )
    loader = UnstructuredFileLoader(file_path)
    docs = loader.load_and_split(text_splitter=splitter)
    embeddings = OpenAIEmbeddings()
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)
    vectorstore = Chroma.from_documents(docs, cached_embeddings)
    retriever = vectorstore.as_retriever()
    return retriever


def send_message(message, role: str, save: bool = True):
    with st.chat_message(role):
        st.markdown(message)
    if save:
        st.session_state["messages"].append({"message": message, "role": role})


def paint_history():
    for message in st.session_state["messages"]:
        send_message(
            message["message"],
            message["role"],
            save=False,
        )


def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)


prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            Answer the question using ONLY the following context. If you don't know the answer just say you don't know. DON'T make anything up.

            Context: {context}
            """,
        ),
        ("human", "{question}"),
    ]
)

st.title("DocumentGPT")
st.markdown(
    """
Welcome!

Use this chatbot to ask questions to an AI about your files!
Upload your files on the sidebar.
"""
)


with st.sidebar:
    openai_key = st.text_input("OpenAI API Key")
    file = st.file_uploader("Upload a .txt file", type="txt")

if file:
    retriever = embed_file(file)
    send_message("I'm ready! Ask away!", "ai", save=False)
    paint_history()

message = st.chat_input("Type a message...", key="message")
if message:
    send_message(message, "human")
    chain = (
        {
            "context": retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
    )
    response = chain.invoke(message)
    send_message(response.content, "ai")
else:
    st.session_state["messages"] = []
