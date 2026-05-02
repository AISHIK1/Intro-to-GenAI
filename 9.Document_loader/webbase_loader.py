
from langchain_community.document_loaders import WebBaseLoader
from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

llm=HuggingFaceEndpoint(
    repo_id='Qwen/Qwen2.5-7B-Instruct',
    task='text-generator',
    temperature=0.5
)

model=ChatHuggingFace(llm=llm)

prompt=PromptTemplate(
    template='Answer the questions /n {questions} for the following text {text}',
    input_variables=['questions','text']
)



url='https://www.buyhawkins.in/ProductDetail.aspx?e=ICG30&f=CKR1'

loader=WebBaseLoader(url)

docs=loader.load()

parser=StrOutputParser()

chain= prompt | model | parser

result=chain.invoke({
    'questions': 'what is the price of the cooker ?',
    'text' : docs[0].page_content
})

print(result)