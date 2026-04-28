from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

docs=[
    "Delhi is the capital city of India.",
    "Paris is the capital city of France.",
    "Tokyo is the capital city of Japan."
]

model=OpenAIEmbeddings(model="text-embedding-3-small",dimensions=32)

result=model.embed_documents(docs)

print(result)