import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import pickle

# Function to load and preprocess data
def load_data(path):
    df = pd.read_csv(Spam.csv)
    df = df[['label', 'message']].dropna()
    df['label'] = df['label'].map({'ham': 0, 'spam': 1})
    return df['message'], df['label']

# Train model and vectorizer
def train_model(messages, labels):
    X_train, X_test, y_train, y_test = train_test_split(
        messages, labels, test_size=0.2, random_state=42, stratify=labels
    )
    vectorizer = TfidfVectorizer(stop_words='english')
    X_train_vect = vectorizer.fit_transform(X_train)
    model = MultinomialNB()
    model.fit(X_train_vect, y_train)
    # Evaluate
    X_test_vect = vectorizer.transform(X_test)
    preds = model.predict(X_test_vect)
    acc = accuracy_score(y_test, preds)
    st.write(f"Model accuracy: {acc:.2%}")
    return model, vectorizer

# Load and train on startup
@st.cache_resource
def load_and_train():
    messages, labels = load_data('spam_dataset.csv')
    return train_model(messages, labels)

model, vectorizer = load_and_train()

# Streamlit interface
st.title('Email/SMS Spam Classifier')

st.markdown('Enter your message below and click **Predict** to check if it is spam or not.')
user_input = st.text_area('Message', height=150)

if st.button('Predict'):
    if not user_input.strip():
        st.warning('Please enter a message to classify.')
    else:
        vect = vectorizer.transform([user_input])
        prediction = model.predict(vect)[0]
        label = 'Spam' if prediction == 1 else 'Not Spam'
        st.success(f'This message is classified as: **{label}**')

# Optional: command to run
# To run the app: streamlit run this_script.py
