# Import necessary libraries
import streamlit as st
import pandas as pd
import numpy as np
import nltk
import string
import time
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

# Download NLTK resources if needed
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

# Preprocessing function for text
def preprocess_text(text):
    """
    Preprocess the input text by:
    1. Removing punctuation,
    2. Tokenizing,
    3. Removing stopwords, and
    4. Stemming words.
    """
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Tokenization
    tokens = nltk.word_tokenize(text)
    
    # Remove stopwords and stem
    stop_words = set(nltk.corpus.stopwords.words('english'))
    ps = nltk.PorterStemmer()
    
    processed_tokens = [ps.stem(word.lower()) for word in tokens if word.lower() not in stop_words]
    
    return ' '.join(processed_tokens)

# ------------------------------
# Set Streamlit Page Config and Background
# ------------------------------
st.set_page_config(
    page_title="Kolkata Weather & Spam Prediction",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Optional: set a background image via custom CSS. Replace URL with your desired link.
def set_bg_image():
    st.markdown(
        """
        <style>
        .stApp {
            background: url("https://raw.githubusercontent.com/username/repository/main/path/to/your_background.jpg");
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
            background-position: center;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

set_bg_image()

# ------------------------------
# Application: Spam Prediction
# ------------------------------

st.title("📧 Spam Prediction Web App")
st.markdown("""
This app allows you to build a spam prediction model using Machine Learning (Naive Bayes and SVM) and 
check if your messages are spam or not.

**Steps:**
1. **Upload a CSV dataset** (with a 'message' and 'label' column where label is 'ham' or 'spam').
2. The app will preprocess the messages (removing punctuation, tokenizing, removing stopwords, and stemming).
3. It will train two models:
   - Naive Bayes (MultinomialNB)
   - Support Vector Machine (SVM with a linear kernel)
4. Evaluation metrics and confusion matrices will be shown.
5. Finally, you can enter a new message to check its prediction.
""")

# Sidebar settings and file uploader:
st.sidebar.header("Model Settings & Data Upload")
upload_file = st.sidebar.file_uploader("Upload CSV Data", type=["csv"])
test_size = st.sidebar.slider("Test Set Size (%)", 10, 40, 20)

# ------------------------------
# Load, Preprocess, and Train
# ------------------------------
if upload_file is not None:
    # Load data into DataFrame
    try:
        df = pd.read_csv(upload_file, sep='\t', header=None, names=['label', 'message'])
    except Exception as e:
        st.error(f"Error reading file: {e}")
        st.stop()
    
    # Map label to numerical values: ham -> 0, spam -> 1
    df['label'] = df['label'].map({'ham': 0, 'spam': 1})
    
    # Preprocess messages
    st.write("Preprocessing messages...")
    df['processed_text'] = df['message'].apply(preprocess_text)
    st.write(df.head())
    
    # Split dataset into training and testing
    X = df['processed_text']
    y = df['label']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size/100, random_state=42, stratify=y
    )
    
    # ---------------------------
    # Build Pipelines for Models
    # ---------------------------
    # Create pipeline for Naive Bayes
    nb_pipeline = make_pipeline(
        TfidfVectorizer(),
        MultinomialNB()
    )
    nb_pipeline.fit(X_train, y_train)
    nb_pred = nb_pipeline.predict(X_test)
    
    # Create pipeline for SVM
    svm_pipeline = make_pipeline(
        TfidfVectorizer(),
        SVC(kernel='linear', probability=True)
    )
    svm_pipeline.fit(X_train, y_train)
    svm_pred = svm_pipeline.predict(X_test)
    
    # ---------------------------
    # Display Results: Naive Bayes
    # ---------------------------
    st.markdown("### Naive Bayes Model Results")
    st.write(f"**Accuracy:** {accuracy_score(y_test, nb_pred):.4f}")
    st.write("**Classification Report:**")
    st.text(classification_report(y_test, nb_pred))
    st.write("**Confusion Matrix:**")
    st.write(confusion_matrix(y_test, nb_pred))
    
    # ---------------------------
    # Display Results: SVM
    # ---------------------------
    st.markdown("### SVM Model Results")
    st.write(f"**Accuracy:** {accuracy_score(y_test, svm_pred):.4f}")
    st.write("**Classification Report:**")
    st.text(classification_report(y_test, svm_pred))
    st.write("**Confusion Matrix:**")
    st.write(confusion_matrix(y_test, svm_pred))
    
    # ---------------------------
    # Interactive Prediction
    # ---------------------------
    st.markdown("### 🔮 Test a Message for Spam/Not Spam")
    user_message = st.text_area("Enter your message below:")
    
    model_choice = st.selectbox("Choose a Model for Prediction", ["Naive Bayes", "SVM"])
    
    if st.button("Predict Message"):
        if user_message.strip() != "":
            # Preprocess input message
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
            st.warning("Please enter a message to predict.")
else:
    st.info("👈 Please upload a CSV file to get started with building the spam prediction model.")

