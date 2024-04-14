from typing import TypedDict

import streamlit as st
from agent import Agent
from pydantic.v1.types import SecretStr


class ChatMessage(TypedDict):
    role: str
    message: str


def send_message(message: str, role: str, save: bool = True):
    with st.chat_message(role):
        st.markdown(message)
    if save:
        st.session_state["messages"].append(ChatMessage(role=role, message=message))


def paint_history():
    message: ChatMessage
    for message in st.session_state.get("messages", []):
        send_message(
            message["message"],
            message["role"],
            save=False,
        )


def main():
    st.set_page_config(
        page_title="SearchGPT",
        page_icon="🔍",
    )
    st.title("SearchGPT")
    st.text("First, input your OpenAI API key in to the sidebar.")
    st.text("Then, search for what you’re curious about!")
    session_openai_key: str | None = st.session_state.get("openai_key")

    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    with st.sidebar:
        st.link_button(
            "Github Repo",
            "https://github.com/flynnpark/fullstack-gpt/tree/assignment19",
        )
        st.session_state["openai_key"] = st.text_input("OpenAI API Key")

    if session_openai_key:
        agent = Agent(openai_key=SecretStr(session_openai_key))
        message = st.chat_input(
            (
                "Type a message..."
                if session_openai_key
                else "Input your OpenAI key in the sidebar first."
            ),
            key="message",
            disabled=not session_openai_key,
        )
        paint_history()
        if message:
            send_message(message, "human")
            result = agent.invoke(message)
            send_message(result, "ai")


if __name__ == "__main__":
    main()
