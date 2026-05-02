from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
from langchain_community.document_loaders import TextLoader
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

llm=HuggingFaceEndpoint(
    repo_id='Qwen/Qwen2.5-7B-Instruct',
    task='text-generation',
    temperature=0.5
)

model=ChatHuggingFace(llm=llm)

loader = TextLoader(
    "C:/Users/aishik.biswas/Documents/GitHub/Intro-to-GenAI/9.Document_loader/cricket.txt",
    encoding="utf-8"
)
docs=loader.load()

#print(docs[0].page_content)
#print(docs[0].metadata)
#print(len(docs))

prompt=PromptTemplate(
    template='Write a summary of the poem - \n {poem}',
    input_variables=['poem']
)

parser=StrOutputParser()

chain=prompt | model | parser

result=chain.invoke({
    'poem':docs[0].page_content
})

print(result)