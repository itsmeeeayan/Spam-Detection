# Import necessary libraries
import streamlit as st
import pandas as pd
import numpy as np
import nltk
import string
import time

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import make_pipeline

# Download NLTK resources if not already downloaded
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

# Preprocessing function with type conversion
def preprocess_text(text):
    # Ensure the input is treated as a string
    if not isinstance(text, str):
        text = str(text)
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Tokenize the text
    tokens = nltk.word_tokenize(text)
    # Remove stopwords and stem the tokens
    stop_words = set(nltk.corpus.stopwords.words('english'))
    ps = nltk.PorterStemmer()
    processed_tokens = [ps.stem(word.lower()) for word in tokens if word.lower() not in stop_words]
    return ' '.join(processed_tokens)

# -------------------------
# Streamlit App for Spam Prediction
# -------------------------
st.set_page_config(
    page_title="Spam Prediction App",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📧 Spam Prediction Web App")
st.markdown("""
This app uses machine learning (Naive Bayes and SVM) to predict if a message is spam or not.  
Steps:
1. Upload your CSV file (in SMSSpamCollection format with columns 'label' and 'message').
2. The app preprocesses the messages and trains models.
3. View the evaluation metrics for both models.
4. Enter your own message to see if it's spam.
""")

# Sidebar for file uploader
st.sidebar.header("Upload Data")
upload_file = st.sidebar.file_uploader("Upload CSV Data", type=["csv"])

if upload_file is not None:
    try:
        # Read CSV file, assuming a tab-separated format, no header
        df = pd.read_csv(upload_file, sep='\t', header=None, names=['label', 'message'])
    except Exception as e:
        st.error(f"Error reading file: {e}")
        st.stop()
    
    # Map the labels to numeric: ham -> 0, spam -> 1
    df['label'] = df['label'].map({'ham': 0, 'spam': 1})
    
    st.write("### Data Sample")
    st.write(df.head())
    
    st.write("Preprocessing messages...")
    # Apply the preprocess_text function (values cast to string internally)
    df['processed_text'] = df['message'].apply(preprocess_text)
    
    # Split dataset into training and testing sets
    X = df['processed_text']
    y = df['label']
    test_size = st.sidebar.slider("Test Set Size (%)", 10, 40, 20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size/100, random_state=42, stratify=y
    )
    
    # ---------------------------
    # Train Naive Bayes Model Pipeline
    # ---------------------------
    st.markdown("### Training Naive Bayes Model")
    nb_pipeline = make_pipeline(
        TfidfVectorizer(),
        MultinomialNB()
    )
    nb_pipeline.fit(X_train, y_train)
    nb_pred = nb_pipeline.predict(X_test)
    
    st.write("**Naive Bayes Results:**")
    st.write(f"Accuracy: {accuracy_score(y_test, nb_pred):.4f}")
    st.text("Classification Report:\n" + classification_report(y_test, nb_pred))
    st.write("Confusion Matrix:")
    st.write(confusion_matrix(y_test, nb_pred))
    
    # ---------------------------
    # Train SVM Model Pipeline
    # ---------------------------
    st.markdown("### Training SVM Model")
    svm_pipeline = make_pipeline(
        TfidfVectorizer(),
        SVC(kernel='linear', probability=True)
    )
    svm_pipeline.fit(X_train, y_train)
    svm_pred = svm_pipeline.predict(X_test)
    
    st.write("**SVM Results:**")
    st.write(f"Accuracy: {accuracy_score(y_test, svm_pred):.4f}")
    st.text("Classification Report:\n" + classification_report(y_test, svm_pred))
    st.write("Confusion Matrix:")
    st.write(confusion_matrix(y_test, svm_pred))
    
    # ---------------------------
    # Prediction Interface
    # ---------------------------
    st.markdown("### 🔮 Test a Message for Spam/Not Spam")
    user_message = st.text_area("Enter your message:")
    model_choice = st.selectbox("Select a Model for Prediction", ["Naive Bayes", "SVM"])
    
    if st.button("Predict Message"):
        if user_message.strip() == "":
            st.warning("Please enter a message to predict.")
        else:
            processed_msg = preprocess_text(user_message)
            if model_choice == "Naive Bayes":
                pred = nb_pipeline.predict([processed_msg])[0]
                prob = nb_pipeline.predict_proba([processed_msg])[0][pred]
            else:
                pred = svm_pipeline.predict([processed_msg])[0]
                prob = svm_pipeline.predict_proba([processed_msg])[0][pred]
            label = "Spam" if pred == 1 else "Not Spam"
            st.success(f"Prediction: **{label}** (Confidence: {prob*100:.1f}%)")
else:
    st.info("👈 Please upload a CSV file to get started with the spam prediction model.")
