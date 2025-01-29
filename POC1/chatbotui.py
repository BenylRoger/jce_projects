# chatbot.py
#pip install sentence-transformers
#pip install transformers langchain faiss-cpu
#pip install streamlit

import os
import streamlit as st
from langchain.document_loaders import TextLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.llms import HuggingFaceHub
from langchain.chains import RetrievalQA
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import FAISS
from langchain_community.chat_models import ChatOllama

# Set up Hugging Face API Key
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "key"

# Load the Document
def load_document(document_path):
    """
    Load the text document for the chatbot's knowledge base.
    """
    print(f"Loading document from: {document_path}")
    return TextLoader(document_path)

# Create a Searchable Knowledge Base
def create_knowledge_base(loader):
    """
    Create a vectorstore-based knowledge base from the loaded document.
    """
    print("Creating a searchable knowledge base...")
    
    # Use SentenceTransformer to generate embeddings
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Create a vector store (FAISS in this case)
    vectorstore = FAISS.from_documents(loader.load(), embeddings)
    return vectorstore.as_retriever()

# Configure the Hugging Face Model
def setup_llm():
    """
    Set up the Hugging Face pre-trained model for the chatbot.
    """
    print("Configuring the Hugging Face model...")
    #return HuggingFaceHub(repo_id="google/flan-t5-base", model_kwargs={"temperature": 0, "max_length": 256})
    return ChatOllama(model="gemma:2b")

# Create the Question-Answering Chain
def create_qa_chain(llm, retriever):
    """
    Combine the LLM and retriever into a RetrievalQA chain.
    """
    print("Setting up the question-answering chain...")
    return RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True)

# Run the Chatbot with Streamlit
def run_chatbot_streamlit(qa_chain):
    """
    Interact with the chatbot through Streamlit.
    """
    st.title("Chatbot with Streamlit")
    st.write("Ask questions based on the provided document.")

    # Input text box for the user to ask questions
    user_query = st.text_input("Enter your question:")
    
    if st.button("Ask"):
        if user_query:
            # Generate a response
            response = qa_chain({"query": user_query})
            answer = response["result"]
            source_docs = response["source_documents"]

            # Display the answer
            st.subheader("Answer:")
            st.write(answer)

            # Display the source documents
            st.subheader("Source(s):")
            for doc in source_docs:
                st.write(f"- {doc.metadata.get('source', 'Unknown source')}")

if __name__ == "__main__":
    # Path to your text document
    document_path = "sample_document.txt"  # Replace with the path to your document

    # Ensure the document exists
    if not os.path.exists(document_path):
        print(f"Error: Document not found at {document_path}")
        exit(1)

    # Load the document
    loader = load_document(document_path)

    # Create the knowledge base
    retriever = create_knowledge_base(loader)

    # Set up the LLM
    llm = setup_llm()

    # Create the QA chain
    qa_chain = create_qa_chain(llm, retriever)

    # Run the chatbot with Streamlit
    run_chatbot_streamlit(qa_chain)
