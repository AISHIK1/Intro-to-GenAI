from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

model=ChatOpenAI(model="gpt-4",temperature=2)

result=model.invoke("Suggest me 5 boys names ")

print(result.content)