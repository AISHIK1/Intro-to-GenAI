import sys

from dotenv import load_dotenv
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
    template="Give a detailed report about \n{topic}",
    input_variables=["topic"],
)

template2 = PromptTemplate(
    template="Give me a 5 lines summary about \n{text}",
    input_variables=["text"],
)

prompt1 = template1.invoke({
    "topic": "Deep learning BERT models",
})

result = model.invoke(prompt1)

prompt2 = template2.invoke({
    "text": result.content,
})

result = model.invoke(prompt2)

print(result.content)
