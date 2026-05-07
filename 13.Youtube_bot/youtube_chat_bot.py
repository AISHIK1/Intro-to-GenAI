from transformers.masking_utils import chunked_overlay
from youtube_transcript_api import YouTubeTranscriptApi,TranscriptsDisabled
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFaceEmbeddings,HuggingFaceEndpoint,ChatHuggingFace
from langchain_core.runnables import RunnableLambda,RunnablePassthrough,RunnableParallel
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

video_id = 'iOdFUJiB0Zc'

try:
    api = YouTubeTranscriptApi()

    transcript_list = api.fetch(video_id)

    transcript = " ".join(
        chunk.text for chunk in transcript_list
    )


except TranscriptsDisabled:
    print('Video not found or transcripts are disabled.')

except Exception as e:
    print(f"An unexpected error occurred: {e}")


def format_word(text):
    return " ".join(docs.page_content for docs in text)

splitter=RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100
)
chunks=splitter.create_documents([transcript])

embeddings=HuggingFaceEmbeddings(
    model_name='BAAI/bge-base-en-v1.5'
)

llm=HuggingFaceEndpoint(
    repo_id='Qwen/Qwen2.5-7B-Instruct',
    task='text-generation',
    temperature=0.2
)

model=ChatHuggingFace(llm=llm)

vector_store=FAISS.from_documents(chunks,embeddings)

retrever=vector_store.as_retriever(search_type='mmr',search_kwargs={'k':10})

prompt=PromptTemplate(
    template=""" You are a helpful AI assistant. Answer to the following questions provided only and only from the comtext provided. If you dnt have enough information about the topic just say 'I dont know'
    \n\n {questions}
    \n\n {context}""",
    input_variables=['questions','context']
)
parser=StrOutputParser()

parallel_chain=RunnableParallel({
    'context': retrever | RunnableLambda(format_word),
    'questions': RunnableParallel()
})

chain= parallel_chain | prompt | model | parser

result=chain.invoke('What is the video about ?')
print(result)