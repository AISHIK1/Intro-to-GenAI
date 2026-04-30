from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint,ChatHuggingFace
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from langchain_core.output_parsers import PydanticOutputParser

load_dotenv()

llm=HuggingFaceEndpoint(
    repo_id='deepseek-ai/DeepSeek-V4-Pro',
    task='text-generation',
    temperature=0.5
)

model=ChatHuggingFace(llm=llm)


class person(BaseModel):
    name:str=Field(description='Name of the person')
    age:int=Field(description="Age of the person")
    city:str=Field(description='City of the person')

parser=PydanticOutputParser(pydantic_object=person)


template=PromptTemplate(
    template='Give me the name, age and city of a Fictional Character {character} \n {format_instruction}',
    input_variables=['character'],
    partial_variables={'format_instruction':parser.get_format_instructions()}

)

chain =template | model | parser

result=chain.invoke({
    'character':'Sherlock Holmes'
})

print(result)
print(type(result))

print(chain.get_graph().draw_ascii())