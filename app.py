import streamlit as st
from langchain.callbacks import StreamingStdOutCallbackHandler
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from controllers import wiki_search
from parsers import JsonOutputParser

session_openai_key = st.session_state.get("openai_key")
st.set_page_config(
    page_title="QuizGPT",
    page_icon="❓",
)
st.title("QuizGPT")
st.markdown(
    """
Welcome!

First, input your OpenAI API key in the sidebar. Then, I will make a quiz from Wikipedia articles or files you upload to test your knowledge and help you study.
"""
)

with st.sidebar:
    docs = None
    st.link_button(
        "Github Repo", "https://github.com/flynnpark/fullstack-gpt/tree/assignment16"
    )
    session_openai_key = st.text_input("OpenAI API Key")
    topic = st.text_input("Search Wikipedia...")
    if topic:
        docs = wiki_search(topic)


message = st.chat_input(
    (
        "Type a message..."
        if session_openai_key
        else "Input your OpenAI key in the sidebar first."
    ),
    key="message",
    disabled=not session_openai_key,
)


llm = (
    ChatOpenAI(
        temperature=0.1,
        model="gpt-3.5-turbo-1106",
        streaming=True,
        callbacks=[StreamingStdOutCallbackHandler()],
    )
    if session_openai_key
    else None
)
output_parser = JsonOutputParser()
questions_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are a helpful assistant that is role playing as a teacher.

Based ONLY on the following context make 10 (TEN) questions to test the user's knowledge about the text.

Each question should have 4 answers, three of them must be incorrect and one should be correct.

Use (O) to signal the correct answer.

Question examples:

Question: What is the color of the ocean?
Answers: Red|Yellow|Green|Blue(O)

Question: What is the capital or Georgia?
Answers: Baku|Tbilisi(O)|Manila|Beirut

Question: When was Avatar released?
Answers: 2007|2001|2009(O)|1998

Question: Who was Julius Caesar?
Answers: A Roman Emperor(O)|Painter|Actor|Model

Your turn!

Context: {context}
""",
        )
    ]
)

formatting_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are a powerful formatting algorithm.

You format exam questions into JSON format.
Answers with (o) are the correct ones.

Example Input:

Question: What is the color of the ocean?
Answers: Red|Yellow|Green|Blue(o)

Question: What is the capital or Georgia?
Answers: Baku|Tbilisi(o)|Manila|Beirut

Question: When was Avatar released?
Answers: 2007|2001|2009(o)|1998

Question: Who was Julius Caesar?
Answers: A Roman Emperor(o)|Painter|Actor|Model


Example Output:

```json
{{
    "questions": [
        {{
            "question": "What is the color of the ocean?",
            "answers": [
                {{
                    "answer": "Red",
                    "correct": false
                }},
                {{
                    "answer": "Yellow",
                    "correct": false
                }},
                {{
                    "answer": "Green",
                    "correct": false
                }},
                {{
                    "answer": "Blue",
                    "correct": true
                }}
            ]
        }},
        {{
            "question": "What is the capital or Georgia?",
            "answers": [
                {{
                    "answer": "Baku",
                    "correct": false
                }},
                {{
                    "answer": "Tbilisi",
                    "correct": true
                }},
                {{
                    "answer": "Manila",
                    "correct": false
                }},
                {{
                    "answer": "Beirut",
                    "correct": false
                }}
            ]
        }},
        {{
            "question": "When was Avatar released?",
            "answers": [
                {{
                    "answer": "2007",
                    "correct": false
                }},
                {{
                    "answer": "2001",
                    "correct": false
                }},
                {{
                    "answer": "2009",
                    "correct": true
                }},
                {{
                    "answer": "1998",
                    "correct": false
                }}
            ]
        }},
        {{
            "question": "Who was Julius Caesar?",
            "answers": [
                {{
                    "answer": "A Roman Emperor",
                    "correct": true
                }},
                {{
                    "answer": "Painter",
                    "correct": false
                }},
                {{
                    "answer": "Actor",
                    "correct": false
                }},
                {{
                    "answer": "Model",
                    "correct": false
                }}
            ]
        }}
    ]
}}
```
Your turn!

Questions: {context}

""",
        )
    ]
)


def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)


@st.cache_data(show_spinner="Making quiz...")
def run_quiz_chain(_docs):
    chain = {"context": questions_chain} | formatting_chain | output_parser
    return chain.invoke(_docs)


if not llm:
    st.stop()

questions_chain = {"context": format_docs} | questions_prompt | llm
formatting_chain = formatting_prompt | llm

if not docs:
    st.stop()


response = run_quiz_chain(docs)
with st.form("questions_form"):
    for question in response["questions"]:
        st.write(question["question"])
        value = st.radio(
            "Select an option.",
            [answer["answer"] for answer in question["answers"]],
            index=None,
        )
        if {"answer": value, "correct": True} in question["answers"]:
            st.success("Correct!")
        elif value is not None:
            st.error("Wrong!")
    button = st.form_submit_button()
