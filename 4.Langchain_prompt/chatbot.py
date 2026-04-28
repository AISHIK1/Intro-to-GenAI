from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage,AIMessage,SystemMessage

load_dotenv()

llm=HuggingFaceEndpoint(
    repo_id='deepseek-ai/DeepSeek-V4-Pro',
    task='textgeneration',
    temperature=0.5)

model=ChatHuggingFace(llm=llm)

chat_history=[
    SystemMessage(content='You are a very helpful and cheerfule AI assistant who always greets before anything else')
]
while(True):
    user_input=input("YOU : ")
    chat_history.append(HumanMessage(content=user_input))
    if(user_input=="exit"):
        break
    result=model.invoke(chat_history)
    chat_history.append(AIMessage(content=result.content))
    print("AI : ",result.content)
print(chat_history)