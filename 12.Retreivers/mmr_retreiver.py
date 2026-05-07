from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
docs=[
    Document(page_content="LangChain makes it easy to work with LLMs."),
    Document(page_content="LangChain is used to build LLM based applications."),
    Document(page_content="Chroma is used to store and search document embeddings."),
    Document(page_content="Embeddings are vector representations of text."),
    Document(page_content="MMR helps you get diverse results when doing similarity search."),
    Document(page_content="LangChain supports Chroma, FAISS, Pinecone, and more."),
]

model=HuggingFaceEmbeddings(
    model='BAAI/bge-base-en-v1.5'
)

vectore_store=FAISS.from_documents(
    documents=docs,
    embedding=model
)

retrever=vectore_store.as_retriever(
    search_type='mmr',
    search_kwargs={'k':3, 'lambda_mult':1}
)

query='What is langchain ?'
result=retrever.invoke(query)

print(" The result is for lambda 1 ")
for i,doc in enumerate(result):
    print(f" The {i+1} result is ===================")
    print(f"continue {doc.page_content}")


retrever2=vectore_store.as_retriever(
    search_type='mmr',
    search_kwargs={'k':3, 'lambda_mult':0.5}
)

result2=retrever2.invoke("What is langchain?")

print(" The result is for lambda 0.5 ")

for i,doc in enumerate(result2):
    print(f" The {i+1} result is ===================")
    print(f"continue {doc.page_content}")

