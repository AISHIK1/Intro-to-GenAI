from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import streamlit as st
from langchain_core.prompts import PromptTemplate, load_prompt

load_dotenv()

model=ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite")

st.header("Research tool")

paper_input=st.selectbox("Select a research paper",["Attention all you need","Bert: deep Bidirectional Transformers","GPT3 L Language Models are Few-Shot Learners"])

style_input=st.selectbox("Select a writing style",["Beginer-Friendly","Code-Based","Mathematical-Based"])

length_input=st.selectbox("Select the length of the summary",["Short : 1-2 paragraphs","Medium : 3-4 paragraphs","Long : 5-10 paragraphs"])



#prompt template
#import the prompt template

template=load_prompt("template.json")






#importing chain
if st.button("Summarize"):
    chain= template | model
    result=chain.invoke(
            {
            'paper_input':paper_input,
            'style_input':style_input,
            'length_input':length_input
            }
    )
    st.write(result.content)