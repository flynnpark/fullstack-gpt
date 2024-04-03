from datetime import datetime

import streamlit as st

today = datetime.today().strftime("%Y-%m-%d %H:%M:%S")
st.title(today)

st.write("Hello, world!")

st.write([1, 2, 3])

st.write({"key": "value"})

model = st.selectbox("Choose a number", ("GPT-3", "GPT-4"))

st.write(f"Selected model: {model}")


name = st.text_input("Enter your name", "John Doe")

st.write(f"Hello, {name}!")
