from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_core.messages import HumanMessage,AIMessage,SystemMessage

load_dotenv()

llm=HuggingFaceEndpoint(
    repo_id='deepseek-ai/DeepSeek-V4-Pro',
    task='textgeneration',
    temperature=0.5
)

model=ChatHuggingFace(llm=llm)

template=ChatPromptTemplate.from_messages([
    ('system','You are very helpful customer support agent'),
    (MessagesPlaceholder(variable_name='chat_history')),
    ('human','{query}')
])

chat_history=[]

with open('4.Langchain_prompt/chat_history.txt','r') as f:
    chat_history.extend(f.readlines())

prompt=template.invoke({
    'chat_history':chat_history,
    'query':'Where is my refund ?'
})

result=model.invoke(prompt)

prompt.append(HumanMessage(content='Where is my refund ?'))
prompt.append(AIMessage(content=result.content))

print(result.content)
print(chat_history)
print(template)
print(prompt)