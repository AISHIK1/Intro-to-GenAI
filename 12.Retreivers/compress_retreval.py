from pydoc import doc

from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import LLMChainExtractor
from langchain_huggingface import HuggingFaceEndpoint,HuggingFaceEmbeddings,ChatHuggingFace
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
from torch import embedding

load_dotenv()

embeddings=HuggingFaceEmbeddings(
    model_name='BAAI/bge-base-en-v1.5'
)

llm=HuggingFaceEndpoint(
    repo_id='deepseek-ai/DeepSeek-V4-Pro',
    temperature=0.5
)

model=ChatHuggingFace(llm=llm)

docs=[
     Document(page_content=(
        """The Grand Canyon is one of the most visited natural wonders in the world.
        Photosynthesis is the process by which green plants convert sunlight into energy.
        Millions of tourists travel to see it every year. The rocks date back millions of years."""
    ), metadata={"source": "Doc1"}),

    Document(page_content=(
        """In medieval Europe, castles were built primarily for defense.
        The chlorophyll in plant cells captures sunlight during photosynthesis.
        Knights wore armor made of metal. Siege weapons were often used to breach castle walls."""
    ), metadata={"source": "Doc2"}),

    Document(page_content=(
        """Basketball was invented by Dr. James Naismith in the late 19th century.
        It was originally played with a soccer ball and peach baskets. NBA is now a global league."""
    ), metadata={"source": "Doc3"}),

    Document(page_content=(
        """The history of cinema began in the late 1800s. Silent films were the earliest form.
        Thomas Edison was among the pioneers. Photosynthesis does not occur in animal cells.
        Modern filmmaking involves complex CGI and sound design."""
    ), metadata={"source": "Doc4"})
]

vector_store=FAISS.from_documents(
    documents=docs,
    embedding=embeddings
)

base_retrever=vector_store.as_retriever(search_kwargs={'k':5})

compressor=LLMChainExtractor.from_llm(model)

compressor_retrever=ContextualCompressionRetriever(
    base_retriever=base_retrever,
    base_compressor=compressor
)

query='What is photosynthesis ?'

result=compressor_retrever.invoke(query)

for doc in result:
    print("======================"*20)
    print(doc.page_content)