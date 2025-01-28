import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma


from langchain_ollama import OllamaEmbeddings


llm = OllamaEmbeddings(model="llama3.2")




document = TextLoader("job_listings.txt").load()
text_splitter= RecursiveCharacterTextSplitter(chunk_size=200,
                                              chunk_overlap=10)
chunks=text_splitter.split_documents(document)
db=Chroma.from_documents(chunks,llm)
retriever = db.as_retriever()

text = input("Enter the query")

docs = retriever.invoke(text)

for doc in docs:
    print(doc.page_content)