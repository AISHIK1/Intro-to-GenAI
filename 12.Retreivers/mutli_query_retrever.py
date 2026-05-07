from importlib import simple

from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint

load_dotenv()

embeddings=HuggingFaceEmbeddings(
    model='BAAI/bge-base-en-v1.5'
)

llm=HuggingFaceEndpoint(
    repo_id='deepseek-ai/DeepSeek-V4-Pro',
    task='text-generation',
    temperature=0.5
)
model=ChatHuggingFace(llm=llm)


docs=[
    Document(page_content="Regular walking boosts heart health and can reduce symptoms of depression.", metadata={"source": "H1"}),
    Document(page_content="Consuming leafy greens and fruits helps detox the body and improve longevity.", metadata={"source": "H2"}),
    Document(page_content="Deep sleep is crucial for cellular repair and emotional regulation.", metadata={"source": "H3"}),
    Document(page_content="Mindfulness and controlled breathing lower cortisol and improve mental clarity.", metadata={"source": "H4"}),
    Document(page_content="Drinking sufficient water throughout the day helps maintain metabolism and energy.", metadata={"source": "H5"}),
    Document(page_content="The solar energy system in modern homes helps balance electricity demand.", metadata={"source": "I1"}),
    Document(page_content="Python balances readability with power, making it a popular system design language.", metadata={"source": "I2"}),
    Document(page_content="Photosynthesis enables plants to produce energy by converting sunlight.", metadata={"source": "I3"}),
    Document(page_content="The 2022 FIFA World Cup was held in Qatar and drew global energy and excitement.", metadata={"source": "I4"}),
    Document(page_content="Black holes bend spacetime and store immense gravitational energy.", metadata={"source": "I5"}),
]

vectore_store=FAISS.from_documents(
    documents=docs,
    embedding=embeddings
)

simple_reteriver=vectore_store.as_retriever(search_type='similarity',
                                            search_kwargs={'k':5})

multi_reteriver=MultiQueryRetriever.from_llm(
    retriever=vectore_store.as_retriever(search_kwargs={'k':5}),
    llm=model
)

query='How to improve energy level and maintain balance ?'

similarity_result=simple_reteriver.invoke(query)

multi_result=multi_reteriver.invoke(query)



for i,doc in enumerate(similarity_result):
    print(f"===================={i+1} Result ============================ \n")
    print(f"{doc.page_content}")
    

for i,doc in enumerate(multi_result):
    print(f"===================={i+1} Result ============================ \n")
    print(f"{doc.page_content}")
    