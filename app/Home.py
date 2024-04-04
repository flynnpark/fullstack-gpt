import streamlit as st

st.set_page_config(
    page_title="FullstackGPT Home",
    page_icon="🧊",
)

st.title("FullstackGPT")

with st.sidebar:
    st.title("Sidebar")
    st.text_input("Enter your name")

tab_one, tab_two, tab_three = st.tabs(["Tab 1", "Tab 2", "Tab 3"])

with tab_one:
    st.title("Tab 1")
    st.write("This is tab 1")

with tab_two:
    st.title("Tab 2")
    st.write("This is tab 2")

with tab_three:
    st.title("Tab 3")
    st.write("This is tab 3")
