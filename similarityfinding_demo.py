import os
from transformers import AutoTokenizer, AutoModel
import numpy as np
import torch

# Load a lightweight Hugging Face model
model_name = "sentence-transformers/all-MiniLM-L6-v2"

# Initialize the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Function to generate embeddings
def embed_query(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
        # Take the mean of the token embeddings as the sentence embedding
        embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings.squeeze().numpy()

# Input texts
text1 = input("Enter the text1: ")
text2 = input("Enter the text2: ")

# Generate embeddings for both texts
response1 = embed_query(text1)
response2 = embed_query(text2)

# Calculate cosine similarity using NumPy dot product
similarity_score = np.dot(response1, response2) / (np.linalg.norm(response1) * np.linalg.norm(response2))

print(f"Similarity Score: {similarity_score * 100:.2f}%")
