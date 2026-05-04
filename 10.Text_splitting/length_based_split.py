from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from sympy import separatevars

loader=PyPDFLoader('10.Text_splitting/dl-curriculum.pdf')

docs=loader.load()

splitter=CharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=0
)

chunks=splitter.split_documents(docs)

print(chunks[0].page_content)