import time

import streamlit as st

st.set_page_config(
    page_title="DocumentGPT",
    page_icon="📄",
    layout="centered",
    initial_sidebar_state="auto",
)
st.title("DocumentGPT")

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
