from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()

llm=HuggingFaceEndpoint(
    repo_id='deepseek-ai/DeepSeek-V4-Pro',
    task='textgeneration',
    temperature=0.5
)

model=ChatHuggingFace(llm=llm)


template=ChatPromptTemplate.from_messages([
    ('system','you are a {domain} epert'),
    ('human','Explain in simple words about the {topic}')
])

prompt=template.invoke({
    'domain':'Indian cricket',
    'topic':'spinner'
})

result=model.invoke(prompt)


print(result.content)
print(prompt)