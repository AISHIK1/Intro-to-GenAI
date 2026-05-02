from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel,RunnableSequence
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

llm=HuggingFaceEndpoint(
    repo_id='Qwen/Qwen2.5-7B-Instruct',
    task='text-genration',
    temperature=0.5
)
model=ChatHuggingFace(llm=llm)

prompt1=PromptTemplate(
    template='Write a post for twitter on topic {topic}',
    input_variables=['topic']
)

prompt2=PromptTemplate(
    template='Write a post for linkedin {topic}',
    input_variables=['topic']
)
parser=StrOutputParser()
chain=RunnableParallel({
    'tweet':RunnableSequence(prompt1,model,parser),
    'linkedin':RunnableSequence(prompt2,model,parser)
}
)
result=chain.invoke({
    'topic':'AI'
})

print("post for Twitter "+"-"*50+"\n",result['tweet'])
print("post for Linkedin "+"-"*50+"\n",result['linkedin'])