import os
from langchain_community.chat_models import ChatOllama
import streamlit as st

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
llm=ChatOllama(model="gemma:2b")

st.title("Ask Anything")

question = st.text_input("Enter the question:")

if question:
    response = llm.invoke(question)
    st.write(response.content)