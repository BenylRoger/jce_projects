import os
from transformers import AutoTokenizer, AutoModel
import torch

# Load a lightweight and valid Hugging Face model
model_name = "sentence-transformers/all-MiniLM-L6-v2"

# Download the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Function to generate embeddings
def embed_query(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
        # Take the mean of the token embeddings as the sentence embedding
        embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings.numpy()

# Input text
text = input("Enter the text: ")

# Generate and print embeddings
response = embed_query(text)
print(response)
