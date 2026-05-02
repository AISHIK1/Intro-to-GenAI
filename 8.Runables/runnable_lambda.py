from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence,RunnableParallel,RunnablePassthrough,RunnableLambda
from dotenv import load_dotenv

load_dotenv()

llm=HuggingFaceEndpoint(
    repo_id='Qwen/Qwen2.5-7B-Instruct',
    task='text-generation',
    temperature=0.5
)

model=ChatHuggingFace(llm=llm)

prompt1=PromptTemplate(
    template="Make a joke on {topic}",
    input_variables=['topic']
)

prompt2=PromptTemplate(
    template='Explain the joke /n {text}',
    input_variables=['text']
)
parser=StrOutputParser()
def word_count(text):
    return len(text.split())



joke_generator=RunnableSequence(prompt1,model,parser)

parallel_chain=RunnableParallel({
    'joke':RunnablePassthrough(),
    'word_count':RunnableLambda(word_count)
})


chain=RunnableSequence(joke_generator,parallel_chain)

result=chain.invoke({
    'topic':'AI'
})
print("Joke Generated "+"-"*50+"\n",result['joke'])
print("word_count "+"-"*50+"\n",result['word_count'])