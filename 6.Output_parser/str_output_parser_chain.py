from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

llm=HuggingFaceEndpoint(
    repo_id='deepseek-ai/DeepSeek-V4-Pro',
    teask='text-generation',
    temperature=0.5
)

model=ChatHuggingFace(llm=llm)

template1=ChatPromptTemplate(
    template='Write a detailed report of the {topic}',
    input_variables=['topic']
)

template2=ChatPromptTemplate(
    template="Write a 5 lines summary of the {text}",
    input_variables=['text']
)

parser=StrOutputParser()

chain=template1 | model| parser | template2 | model
result=chain.invoke({
    'topic':'deep learning models Bert'
}
)

print(result)