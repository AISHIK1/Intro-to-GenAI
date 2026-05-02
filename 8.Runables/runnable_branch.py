from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence,RunnableParallel,RunnablePassthrough,RunnableLambda,RunnableBranch
from dotenv import load_dotenv

load_dotenv()

llm=HuggingFaceEndpoint(
    repo_id='Qwen/Qwen2.5-7B-Instruct',
    task='text-generation',
    temperature=0.5
)

model=ChatHuggingFace(llm=llm)

prompt1=PromptTemplate(
    template="Generate a detailed report on {topic}",
    input_variables=['topic']
)

prompt2=PromptTemplate(
    template='Summarize the text in less than 100 words /n {text}',
    input_variables=['text']
)
parser=StrOutputParser()

report_generator=RunnableSequence(prompt1,model,parser)

condition_chain=RunnableBranch(
    (lambda x: len(x.split())>300,RunnableSequence(prompt2,model,parser)),
    RunnablePassthrough()
)

chain=RunnableSequence(report_generator,condition_chain)

result=chain.invoke({
    'topic':'Russia vs Ukraine War'
})

print(result)