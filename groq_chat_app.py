import streamlit as st
import os 
from groq import Groq
import random 

from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate

from dotenv import load_dotenv
load_dotenv()

def main():
    groq_api_key = os.getenv("GROQ_API_KEY")

    # Display the groq logo 
    # spacer, col = st.columns([5, 1])
    # with col: 
    #     st.image("groqcloud_darkmode.png")

    
    st.title("Groq Chat App")
    st.write("This is a simple chat app built with Streamlit and Groq's API. It uses Langchain to manage the conversation flow and memory.")

    st.sidebar.title("Customization")
    model = st.sidebar.selectbox("Select a model", ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "openai/gpt-oss-120b"])

    conversation_memory_length = st.sidebar.slider("Conversation Memory Length", min_value=1, max_value=10, value=5)
    memory = ConversationBufferMemory(k=conversation_memory_length)
    user_question = st.text_input("Ask a question to the Groq model:")

    # session state variable 
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    else: 
        for message in st.session_state.chat_history:
            memory.save_context({"input": message['human']}, {"output": message['AI']})


    # Initialize the Groq client and Langchain GroqChat wrapper
    llm = ChatGroq(groq_api_key=groq_api_key, model=model)
    conversation = ConversationChain(llm=llm, memory=memory) 

    # if the user has asked a question, get the response from the model and display it
    if user_question:
        response = conversation.run(user_question)
        message = {"human": user_question, "AI": response}
        st.session_state.chat_history.append(message)
        st.write(f"**Groq:** {response}")

if __name__ == "__main__":
    main()