import os
from langchain_community.llms import Ollama
import streamlit as st
from langchain.prompts import PromptTemplate

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
#llm = Ollama(model="llama3.2:1b")
llm = Ollama(model="phi")
prompt_template = PromptTemplate(
    input_variables=["city","month","language","budget"],
    template="""Welcome to the {city} travel guide!
    If you're visiting in {month}, here's what you can do:
    1. Must-visit attractions.
    2. Local cuisine you must try.
    3. Useful phrases in {language}.
    4. Tips for traveling on a {budget} budget.
    Enjoy your trip!
    """
)

st.title("Travel Guide")

city = st.text_input("Enter the city:")
month = st.text_input("Enter the month of travel")
language = st.text_input("Enter the language:")
budget = st.selectbox("Travel Budget",["Low","Medium","High"])

chain = prompt_template | llm

if city and month and language and budget:
    response = chain.invoke({"city":city,
                             "month":month,
                             "language":language,
                             "budget":budget
                             })
    st.write(response)