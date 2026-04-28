from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

load_dotenv()

model=OpenAIEmbeddings(model="text-embedding-3-small",dimensions=300)

document=[
    "Virat Kohli is an Indian cricketer known for his aggressive batting and leadership.",
    "MS Dhoni is a former Indian captain famous for his calm demeanor and finishing skills.",
    "Sachin Tendulkar, also known as the 'God of Cricket', holds many batting records.",
    "Rohit Sharma is known for his elegant batting and record-breaking double centuries.",
    "Jasprit Bumrah is an Indian fast bowler known for his unorthodox action and yorkers."
]

query="Who is the captain of India ?"

doc_embedding=model.embed_documents(document)
query_embedding=model.embed_query(query)

scores=cosine_similarity([query_embedding],doc_embedding)[0]

index,scores=sorted(list(enumerate(scores)),key=lambda x:x[1])[-1]
print(f"index document :{index}, and the score : {scores}")

print(f"query : {query}")
print( f"{document[index]}")