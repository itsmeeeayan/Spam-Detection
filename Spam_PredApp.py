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

# Preprocessing function for text
def preprocess_text(text):
    # Ensure the input is a string
    if not isinstance(text, str):
        text = str(text)
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Tokenize text
    tokens = nltk.word_tokenize(text)
    # Remove stopwords and stem
    stop_words = set(nltk.corpus.stopwords.words('english'))
    ps = nltk.PorterStemmer()
    processed_tokens = [ps.stem(word.lower()) for word in tokens if word.lower() not in stop_words]
    return ' '.join(processed_tokens)

# Set page configuration (must come before any other Streamlit calls)
st.set_page_config(
    page_title="Spam Prediction App",
    layout="wide",
    initial_sidebar_state="expanded"
)

# (Optional) Set a background image using CSS (update the URL to your own GitHub raw image URL)
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

# App Title and Description
st.title("📧 Spam Prediction Web App")
st.markdown("""
This app uses machine learning (Naive Bayes and SVM) to predict if a message is spam or not.
**Steps:**
1. Upload your CSV file (in SMSSpamCollection format with columns 'label' and 'message').
2. The app preprocesses the messages (removes punctuation, tokenizes, removes stopwords, and stems).
3. It trains two models:
   - Naive Bayes (MultinomialNB)
   - Support Vector Machine (SVM with a linear kernel)
4. Evaluation metrics and confusion matrices are displayed.
5. Finally, you can enter a new message to check whether it's spam.
""")

# Sidebar for file uploader and settings
st.sidebar.header("Model Settings & Data Upload")
upload_file = st.sidebar.file_uploader("Upload CSV Data", type=["csv"])
test_size = st.sidebar.slider("Test Set Size (%)", 10, 40, 20)

if upload_file is not None:
    try:
        # Read CSV file; for SMSSpamCollection, it's tab-separated with no header
        df = pd.read_csv(upload_file, sep='\t', header=None, names=['label', 'message'])
    except Exception as e:
        st.error(f"Error reading file: {e}")
        st.stop()
    
    # Drop rows with missing values in 'message' or 'label'
    df = df.dropna(subset=['message', 'label'])
    
    # Map labels: ham -> 0, spam -> 1
    df['label'] = df['label'].map({'ham': 0, 'spam': 1})
    
    st.write("### Data Sample")
    st.write(df.head())
    
    st.write("Preprocessing messages...")
    df['processed_text'] = df['message'].apply(preprocess_text)
    
    # Split dataset
    X = df['processed_text']
    y = df['label']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size/100, random_state=42, stratify=y
    )
    
    # ---------------------------
    # Build Pipelines for Models
    # ---------------------------
    nb_pipeline = make_pipeline(
        TfidfVectorizer(),
        MultinomialNB()
    )
    nb_pipeline.fit(X_train, y_train)
    nb_pred = nb_pipeline.predict(X_test)
    
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
