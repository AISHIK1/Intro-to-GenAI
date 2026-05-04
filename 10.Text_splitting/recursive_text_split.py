from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from dotenv import load_dotenv

load_dotenv()

loader = PyPDFLoader('10.Text_splitting/dl-curriculum.pdf')
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=0
)

chunks = splitter.split_documents(docs)

print(chunks[0].page_content)  