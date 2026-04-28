from langchain_core.messages import HumanMessage,AIMessage,SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

model=ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite")

message=[
    SystemMessage(content="you are a helpful and cheerfull AI assistant"),
    HumanMessage(content="what is the capital city of India ?"),
]

result=model.invoke(message)
print(result.content)
message.append(AIMessage(content=result.content))
print(message)