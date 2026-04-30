from dotenv import load_dotenv
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from pydantic import BaseModel, Field


class TopicFacts(BaseModel):
    fact_1: str = Field(description="First fact about the topic")
    fact_2: str = Field(description="Second fact about the topic")
    fact_3: str = Field(description="Third fact about the topic")
    fact_4: str = Field(description="Fourth fact about the topic")

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-V4-Pro",
    task="conversational",
    temperature=0.5,
)

model = ChatHuggingFace(llm=llm)

parser = PydanticOutputParser(pydantic_object=TopicFacts)

template = PromptTemplate(
    template="Give me 4 facts about {topic}.\n{format_instructions}",
    input_variables=["topic"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

chain = template | model | parser

result = chain.invoke({"topic": "artificial intelligence"})

print(result)
