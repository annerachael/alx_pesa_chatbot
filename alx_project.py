import streamlit as st
import openai
import pandas as pd
from dotenv import load_dotenv
import os

load_dotenv()
# Initialize OpenAI API


openai.api_key = os.environ.get("openai.api_key") 
# Initialize chat history in session state
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Streamlit App
st.title("Financial Data Q&A with RAG")

st.sidebar.header("Upload your CSV file")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")

if uploaded_file:
    # Load the entire CSV content
    df = pd.read_csv(uploaded_file)
    st.write("CSV Data Preview:", df)


    # Function to retrieve relevant data for the query
    def retrieve_data(question):
        relevant_data = df.to_string(index=False)
        return relevant_data


    # Display chat history
    if st.session_state.messages:
        for message in st.session_state.messages:
            role, content = message["role"], message["content"]
            if role == "user":
                st.write(f"**User:** {content}")
            else:
                st.write(f"**Assistant:** {content}")

    # Text area to input questions
    question = st.text_area("Ask a question about the financial data:")

    if st.button("Get Answer"):
        # Add user question to chat history
        st.session_state.messages.append({"role": "user", "content": question})

        # Retrieve relevant data based on the question
        relevant_data = retrieve_data(question)

        # Use OpenAI ChatCompletion to generate an answer with RAG approach
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": """Instructions:
                1. Use the dataset to answer the questions.
                2. Ensure the answers are accurate and based on the data provided.
                3. Answers should be in the requested format, for example decimal, percentage, etc.
                4. Summarize or conclude the answer to avoid incomplete responses.
                Do not answer questions unrelated to the Financial document uploaded.
                Give accurate resposes only"""},
                {"role": "user",
                 "content": f"The following is financial data in tabular format:\n{relevant_data}\n\nAnswer the question: {question}"}
            ],
            max_tokens=300
        )

        # Extract the answer
        answer = response['choices'][0]['message']['content'].strip()

        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": answer})

        # Display the assistant's answer
        st.write("**Assistant:**", answer)
