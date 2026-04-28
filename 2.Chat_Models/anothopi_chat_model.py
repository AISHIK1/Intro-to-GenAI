from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv

load_dotenv()

model=ChatAnthropic(model="claude-2",temperature=0.5)

result=modek.invoke("Suggest me 5 names for boys ")

print(result.content)