import os
import json
import datetime
import csv
import nltk
import ssl
import streamlit as st
import random
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

ssl._create_default_https_context = ssl._create_unverified_context
nltk.data.path.append(os.path.abspath("nltk_data"))
nltk.download('punkt')

# Load intents from the JSON file
file_path = os.path.abspath("intents.json")
with open(file_path, "r") as file:
    intents = json.load(file)

# Initialize the vectorizer and classifier
vectorizer = TfidfVectorizer(ngram_range=(1, 4))
clf = LogisticRegression(random_state=0, max_iter=10000)

# Prepare the data for training
tags = []
patterns = []
for intent in intents:
    for pattern in intent['patterns']:
        tags.append(intent['tag'])
        patterns.append(pattern)

# Train the model
x = vectorizer.fit_transform(patterns)
y = tags
clf.fit(x, y)

def evaluate_expression(input_text):
    # Identify and evaluate mathematical expressions
    math_pattern = r'[-+]?\d*\.\d+|[-+]?\d+|[+\-*/()]'
    tokens = re.findall(math_pattern, input_text.replace(" ", ""))
    
    if tokens:
        try:
            expression = "".join(tokens)
            result = eval(expression)  # Caution: Consider a safer alternative for production
            return str(result)
        except Exception:
            return "Sorry, I couldn't evaluate that expression."
    
    return None

def chatbot(input_text):
    # First check for arithmetic expressions
    result = evaluate_expression(input_text)
    if result is not None:
        return result

    # Process intents if no arithmetic expression found
    input_text = vectorizer.transform([input_text])
    tag = clf.predict(input_text)[0]
    for intent in intents:
        if intent['tag'] == tag:
            response = random.choice(intent['responses'])
            return response

counter = 0

def main():
    global counter
    st.title("Intents of Chatbot using NLP")

    # Sidebar menu options
    menu = ["Home", "Conversation History", "About"]
    choice = st.sidebar.selectbox("Menu", menu)

    # Home Menu
    if choice == "Home":
        st.write("Welcome to the chatbot. Please type a message to get started.")

        # Initialize chat log if it doesn't exist
        if not os.path.exists('chat_log.csv'):
            with open('chat_log.csv', 'w', newline='', encoding='utf-8') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow(['User Input', 'Chatbot Response', 'Timestamp'])

        counter += 1
        user_input = st.text_input("You:", key=f"user_input_{counter}")

        if user_input:
            user_input_str = str(user_input)

            response = chatbot(user_input)
            st.text_area("Chatbot:", value=response, height=120, max_chars=None, key=f"chatbot_response_{counter}")

            # Record the current timestamp
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # Log the interaction to the CSV file
            with open('chat_log.csv', 'a', newline='', encoding='utf-8') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow([user_input_str, response, timestamp])

            if response.lower() in ['goodbye', 'bye']:
                st.write("Thank you for chatting! Have a great day!")
                st.stop()

    # Conversation History Menu
    elif choice == "Conversation History":
        st.header("Conversation History")
        with open('chat_log.csv', 'r', encoding='utf-8') as csvfile:
            csv_reader = csv.reader(csvfile)
            next(csv_reader)  # Skip the header row
            for row in csv_reader:
                st.text(f"User: {row[0]}")
                st.text(f"Chatbot: {row[1]}")
                st.text(f"Timestamp: {row[2]}")
                st.markdown("---")

    elif choice == "About":
        st.write("This project aims to develop a chatbot that comprehends and responds to user inquiries based on predefined intents. The chatbot utilizes Natural Language Processing (NLP) techniques and Logistic Regression to interpret user input effectively.")

        st.subheader("Project Overview:")
        st.write("""
        The project consists of two main components:
        1. The chatbot is trained using NLP methods and a Logistic Regression model on a set of labeled intents.
        2. Streamlit is used to create a user-friendly web interface for the chatbot, allowing users to interact seamlessly.
        """)

        st.subheader("Dataset:")
        st.write("""
        The dataset comprises a collection of labeled intents and corresponding entities. It is structured as follows:
        - Intents: Categories representing user intent (e.g., "greeting", "budget", "about").
        - Entities: Specific phrases extracted from user input (e.g., "Hi", "How do I create a budget?", "What is your purpose?").
        - Text: The actual user input.
        """)

        st.subheader("Streamlit Chatbot Interface:")
        st.write("The chatbot interface is designed using Streamlit, featuring a text input box for user queries and a display area for chatbot responses. The trained model generates replies based on user input.")

        st.subheader("Conclusion:")
        st.write("This project successfully creates a chatbot capable of understanding and responding to user inquiries based on predefined intents. By leveraging NLP and Logistic Regression, along with Streamlit for the interface, the chatbot can be further developed with additional data and advanced NLP techniques.")

if __name__ == '__main__':
    main()
