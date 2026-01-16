import streamlit as st
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms import Ollama



import os 
from dotenv import load_dotenv
load_dotenv()


## Langsmith Tracking 

os.environ['LANGCHAIN_API_KEY'] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Simple QnA Chatbot with Ollama"


##Prompt Template

prompt = ChatPromptTemplate(
    
    [
        ("system","You are a helpful assistant.Please respond to the user queries"),
        ("user","Question:{question}")
    ]
)


def generate_response(question, llm_name, temperature, max_tokens):
    # Pass the slider values into the model here
    llm = Ollama(
        model=llm_name, 
        temperature=temperature,
        # Ollama uses 'num_predict' for max tokens, not 'max_tokens'
        num_predict=max_tokens 
    )
    output_parser = StrOutputParser()
    chain = prompt | llm | output_parser
    answer = chain.invoke({'question': question})
    return answer
    
## Title of the app
st.title("Enhanced QnA chatbot")


## Drop down to select various models

llm = st.sidebar.selectbox("Select a model",["llama2"])

## Adjust response parameter
temperature = st.sidebar.slider("Temperature",min_value = 0.0,max_value =1.0,value = 0.7)
max_tokens = st.sidebar.slider("Max Tokens",min_value = 50,max_value= 300,value=150)


## Main interface for user input

st.write("Go ahead and ask any question")
user_input = st.text_input("You:")

if user_input:
    response = generate_response(user_input,llm,temperature,max_tokens)
    st.write(response)
    
else:
    st.write("Please provide the queries")
    
    