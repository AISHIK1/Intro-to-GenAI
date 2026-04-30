from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


load_dotenv()

llm=HuggingFaceEndpoint(
    repo_id='google/gemma-4-31B-it',
    task='text-generation',
    temperature=0.5
)

model=ChatHuggingFace(llm=llm)

parser=StrOutputParser()

template1= PromptTemplate(
    template='Give me a detailed explanation of {topic}',
    input_variables=['topic']
)

template2=PromptTemplate(
    input_variables=['text'],
    template='Give me 5 points summary of the {text}'
    
)

chain=template1 | model | parser | template2 | model | parser

result=chain.invoke({
    'topic':'dertivates'
})

print(result)