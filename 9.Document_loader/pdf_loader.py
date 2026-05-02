from langchain_community.document_loaders import PyPDFLoader

loader=PyPDFLoader(r"C:\Users\aishik.biswas\Documents\GitHub\Intro-to-GenAI\9.Document_loader\books\mml-book.pdf")

docs=loader.load()

print(len(docs))
print(docs[0].page_content)
print(docs[0].metadata)