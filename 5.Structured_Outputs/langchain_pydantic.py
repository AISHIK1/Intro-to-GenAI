from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from typing import TypedDict,Annotated,Optional,Literal
from pydantic import BaseModel,Field
load_dotenv()

model=ChatOpenAI(model="gpt-3.5-turbo",temperature=0.5)

#schema

class review(BaseModel):
    key_themes:list[str]=Field(description="Mention all the key themes that are discussed in the review")

    summary:str=Field(description='Provide a concise summary of the review in 2-3 sentences')

    sentiment:Literal['Pos','Neg']=Field(description='Classify the overall sentiment of the review as Positive, Negative, or Neutral')

    pros:Optional[list[str]]=Field(description='Write down the pros in points, if you cant find any pros write None')

    cons:Optional[list[str]]=Field(description='Write down the cons in points, if you cant find any cons write None')

    reviewer:str=Field(description='Mention the name of the reviewer if there is not write its annonymous')


    

structured_output=model.with_structured_output(review)

result=structured_output.invoke("""I recently upgraded to the Samsung Galaxy S24 Ultra, and I must say, it’s an absolute powerhouse! The Snapdragon 8 Gen 3 processor makes everything lightning fast—whether I’m gaming, multitasking, or editing photos. The 5000mAh battery easily lasts a full day even with heavy use, and the 45W fast charging is a lifesaver.

The S-Pen integration is a great touch for note-taking and quick sketches, though I don't use it often. What really blew me away is the 200MP camera—the night mode is stunning, capturing crisp, vibrant images even in low light. Zooming up to 100x actually works well for distant objects, but anything beyond 30x loses quality.

However, the weight and size make it a bit uncomfortable for one-handed use. Also, Samsung’s One UI still comes with bloatware—why do I need five different Samsung apps for things Google already provides? The $1,300 price tag is also a hard pill to swallow.

Pros:
Insanely powerful processor (great for gaming and productivity)
Stunning 200MP camera with incredible zoom capabilities
Long battery life with fast charging
S-Pen support is unique and useful 
                                
                                
Cons:
Heavy and bulky design
Bloatware from Samsung is annoying
Expensive price point
Overall, the Galaxy S24 Ultra is a fantastic device fusers and photography enthusiasts, but it may not be choice for those who prioritize portability or value. Ifafford it and want top-tier performance, it’s definiteconsidering. Just be prepared for the size and price!
                                
                            reviewed by Aishik Biswas   """)

print(result)
print("\n Key Themes",result.key_themes)
print("\n Summary",result.summary)
print("\n Sentiment",result.sentiment)
print("\n Pros",result.pros)
print("\n Cons",result.cons)
print("\n Reviewer",result.reviewer)