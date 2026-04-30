from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate 


load_dotenv()

llm=HuggingFaceEndpoint(
    repo_id='google/gemma-4-31B-it',
    task='text-generation',
    temperature=0.5
)

model=ChatHuggingFace(llm=llm)

parser=StrOutputParser()

template=PromptTemplate(
    template='Give me a 5 interesting facts about {topic}',
    input_variables=['topic']
)

chain=template | model | parser

result=chain.invoke({
    'topic':'indian Museum'
})

print(result)
