from langchain_huggingface import HuggingFaceEmbeddings

embeddings=HuggingFaceEmbeddings(model="sentence-transformers/all-MiniLM-L6-v2")

docs=[
    "Delhi is the capital city of India.",
    "Paris is the capital city of France.",
    "Tokyo is the capital city of Japan."
]

vector=embeddings.embed_documents(docs)

print(vector)