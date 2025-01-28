import os
from langchain_ollama import OllamaEmbeddings
import numpy as np

llm = OllamaEmbeddings(model="llama3.2")

# Input texts
text1 = input("Enter the text1: ")
text2 = input("Enter the text2: ")

# Generate embeddings for both texts
response1 = llm.embed_query(text1)
response2 = llm.embed_query(text2)

# Calculate cosine similarity using NumPy dot product
similarity_score = np.dot(response1, response2) / (np.linalg.norm(response1) * np.linalg.norm(response2))

print(f"Similarity Score: {similarity_score * 100:.2f}%")