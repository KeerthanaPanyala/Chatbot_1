from langchain_experimental.agents.agent_toolkits.csv.base import create_csv_agent
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv
import os
import streamlit as st
import pandas as pd
from io import StringIO

def main():
    load_dotenv()

    # Load the OpenAI API key from the environment variable
    if st.secrets["OPENAI_API_KEY"] is None or st.secrets["OPENAI_API_KEY"] == "":
        st.error("OPENAI_API_KEY is not set")
        return

    st.sidebar.title("Navigation")
    pages = ["Upload Dataset", "Ask Questions"]
    selection = st.sidebar.radio("Go to", pages)

    if selection == "Upload Dataset":
        upload_dataset()
    elif selection == "Ask Questions":
        ask_questions()

def upload_dataset():
    st.header("Upload Dataset")
    csv_file = st.file_uploader("Upload a CSV file", type="csv")

    if csv_file is not None:
        # Read the CSV file into a DataFrame
        df = pd.read_csv(csv_file)
        # Save the DataFrame in session state
        st.session_state['dataframe'] = df
        st.session_state['chat_history'] = []  # Initialize chat history
        st.success("CSV file uploaded successfully!")

def ask_questions():
    st.header("Ask Questions")
    
    # Check if a DataFrame is already uploaded
    if 'dataframe' in st.session_state:
        df = st.session_state['dataframe']
        user_question = st.text_input("Ask a question about your CSV:")

        if st.button("Submit"):
            if user_question:
                with st.spinner("In progress..."):
                    # Convert the DataFrame to a CSV string for the agent
                    csv_string = StringIO()
                    df.to_csv(csv_string, index=False)
                    csv_string.seek(0)
                    
                    # Create the CSV agent with the allow_dangerous_code parameter
                    agent = create_csv_agent(
                        ChatOpenAI(temperature=0, model="gpt-3.5-turbo"), 
                        csv_string, 
                        verbose=True, 
                        handle_parsing_errors=True,
                        allow_dangerous_code=True  # Opt-in for dangerous code execution
                    )
                    
                    # Prepare chat history context
                    chat_history_context = "\n".join(
                        [f"Q: {q['question']}\nA: {q['answer']}" for q in st.session_state['chat_history']]
                    )
                    # Ask the question with history context
                    full_question = f"{chat_history_context}\nQ: {user_question}\nA:"
                    answer = agent.run(full_question)
                    
                    # Store the question and answer in the chat history
                    st.session_state['chat_history'].append({"question": user_question, "answer": answer})
                    
                    # Display the chat history
                    for chat in st.session_state['chat_history']:
                        st.write(f"**Q:** {chat['question']}")
                        st.write(f"**A:** {chat['answer']}")
                        st.write("---")  # Separator
            else:
                st.warning("Please enter a question.")
    else:
        st.warning("Please upload a CSV file first in the 'Upload Dataset' section.")

if __name__ == "__main__":
    main()
