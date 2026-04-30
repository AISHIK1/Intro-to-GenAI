import sys

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

sys.stdout.reconfigure(encoding="utf-8")

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-V4-Pro",
    task="conversational",
    temperature=0.5,
)

model = ChatHuggingFace(llm=llm)

template1 = PromptTemplate(
    template="Write a detailed report of the {topic}",
    input_variables=["topic"],
)

template2 = PromptTemplate(
    template="Write a 5 lines summary of the {text}",
    input_variables=["text"],
)

parser = StrOutputParser()

report_chain = template1 | model | parser
summary_chain = template2 | model | parser

report = report_chain.invoke({
    "topic": "deep learning models BERT",
})

result = summary_chain.invoke({
    "text": report,
})

print(result)
