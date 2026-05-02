from langchain_community.document_loaders import DirectoryLoader,PyPDFLoader

loader = DirectoryLoader(
    path="C:/Users/aishik.biswas/Documents/GitHub/Intro-to-GenAI/9.Document_loader/books/",
    glob="*.pdf",
    loader_cls=PyPDFLoader
)


docs=loader.lazy_load()
#docs=loader.load()

for documents in docs:
    print(documents.metadata)
print(len(docs))

