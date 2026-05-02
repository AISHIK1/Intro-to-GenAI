from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableBranch,RunnableParallel,RunnableLambda
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from pydantic import BaseModel,Field
from typing import Literal
from dotenv import load_dotenv

load_dotenv()

class feedback(BaseModel):

    sentiment: Literal['positive','negetive']=Field(description='Give the sentiment of the feedback')

parser=PydanticOutputParser(pydantic_object=feedback)

llm=HuggingFaceEndpoint(
    repo_id='google/gemma-4-31B-it',
    task='text-generation',
    temperature=0.5
)

model=ChatHuggingFace(llm=llm)

prompt1=PromptTemplate(
    template='Classify the sentiment of the feedback into positive or negtive /n {text} /n {format_instruction}',
    input_variables=['text'],
    partial_variables={'format_instruction':parser.get_format_instructions()}
)

prompt2=PromptTemplate(
    template='write a appopriate response for the customer for the postive feedabck /n {feedback}',
input_variables=['feedback']
)

prompt3=PromptTemplate(
    template='write a appopriate response for the customer for the negetive feedback /n {feedback}',
input_variables=['feedback']
)

parser2=StrOutputParser()

classifier_chain=prompt1 | model | parser
branch_chain=RunnableBranch(
    (lambda x:x.sentiment =='negetive' , prompt3 | model | parser2),
    (lambda x:x.sentiment =='positive', prompt2 | model | parser2),
    RunnableLambda(lambda x: 'Could not find sentiment')
)

chain=classifier_chain | branch_chain

result=chain.invoke({
    'text': 'The smartphone is shit'
}
)

print(result)