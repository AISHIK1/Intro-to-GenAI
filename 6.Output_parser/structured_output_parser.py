from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain.output_parsers import StructuredOutputParser
from langchain.output_parsers import ResponseSchema

load_dotenv()

llm=HuggingFaceEndpoint(
    repo_id='deepseek-ai/DeepSeek-V4-Pro',
    task='text-generation',
    temperature=0.5
)

model=ChatHuggingFace(llm=llm)

parser=StructuredOutputParser()

schema=[
    ResponseSchema(name='Fact 1',description='Fact 1 about the topic'),
    ResponseSchema(name='Fact 2',description='Fact 2 about the topic'),
    ResponseSchema(name='Fact 3',description='Fact 3 about the topic'),
    ResponseSchema(name='Fact 4',description='Fact 4 about the topic'),
]

parser=StructuredOutputParser.from_response_schema()

template=PromptTemplate(
    template='Give me 4 facts about {topic} \n {format_instruction}',
    input_variables=[],
    partial_variables={'format_instruction':parser.get_format_instrcutions()}
)

chain=template | model | parser

result=chain.invoke({})

print(result)