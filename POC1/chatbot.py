# chatbot.py
#pip install sentence-transformers
#pip install transformers langchain faiss-cpu

import os
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

# Run the Chatbot
def run_chatbot(qa_chain):
    """
    Interact with the chatbot through the terminal.
    """
    print("\nChatbot is ready! Ask your questions based on the provided document.")
    print("Type 'exit' to quit.\n")

    while True:
        query = input("You: ")
        if query.lower() == "exit":
            print("Goodbye!")
            break

        # Generate a response
        response = qa_chain({"query": query})
        answer = response["result"]
        source_docs = response["source_documents"]

        # Display the answer
        print("\nChatbot:")
        print(answer)

        # Show the source documents
        print("\nSource(s):")
        for doc in source_docs:
            print(f"- {doc.metadata.get('source', 'Unknown source')}")

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

    # Run the chatbot
    run_chatbot(qa_chain)
