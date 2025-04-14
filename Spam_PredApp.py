# Import necessary libraries
import streamlit as st
import pandas as pd
import numpy as np
import nltk
import string

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import make_pipeline

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Preprocessing function for text
def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = nltk.word_tokenize(text)
    stop_words = set(nltk.corpus.stopwords.words('english'))
    ps = nltk.PorterStemmer()
    return ' '.join([ps.stem(word.lower()) for word in tokens if word.lower() not in stop_words])

# Streamlit page configuration
st.set_page_config(page_title="Spam Prediction App", layout="wide")

# Optional: Set background image
def set_bg_image():
    st.markdown(
        """
        <style>
        .stApp {
            background: url("https://raw.githubusercontent.com/username/repository/main/path/to/your_background.jpg");
            background-size: cover;
        }
        </style>
        """, unsafe_allow_html=True
    )

# Uncomment to use background image
# set_bg_image()

st.title("📧 Spam Prediction Web App")
st.markdown("""
This app predicts whether a message is **spam** or **not spam** using ML algorithms like Naive Bayes and SVM.
""")

# Sidebar upload and test size
st.sidebar.header("Upload Data & Settings")
upload_file = st.sidebar.file_uploader("Upload a CSV file (columns: label, message)", type=["csv"])
test_size = st.sidebar.slider("Test Set Size (%)", 10, 40, 20)

if upload_file:
    try:
        df = pd.read_csv(upload_file, sep='\t', header=None, names=['label', 'message'])
    except Exception:
        df = pd.read_csv(upload_file)

    # Sanity checks
    if 'label' not in df.columns or 'message' not in df.columns:
        st.error("CSV must contain 'label' and 'message' columns.")
        st.stop()

    df.dropna(subset=['label', 'message'], inplace=True)

    df['label'] = df['label'].map({'ham': 0, 'spam': 1})
    df.dropna(inplace=True)  # Drop if any label conversion failed

    st.write("### Sample Data", df.head())

    df['processed_text'] = df['message'].apply(preprocess_text)

    X = df['processed_text']
    y = df['label']

    # Check data size
    if len(X) < 5:
        st.error("Insufficient data after preprocessing. Upload a larger dataset.")
        st.stop()

    # Safe split with stratify check
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size / 100, stratify=y, random_state=42
        )
    except ValueError as e:
        st.error(f"Train-test split failed: {e}")
        st.stop()

    # Build and train Naive Bayes pipeline
    nb_pipeline = make_pipeline(TfidfVectorizer(), MultinomialNB())
    nb_pipeline.fit(X_train, y_train)
    nb_pred = nb_pipeline.predict(X_test)

    # Build and train SVM pipeline
    svm_pipeline = make_pipeline(TfidfVectorizer(), SVC(kernel='linear', probability=True))
    svm_pipeline.fit(X_train, y_train)
    svm_pred = svm_pipeline.predict(X_test)

    # Display results
    st.subheader("Naive Bayes Results")
    st.write(f"Accuracy: {accuracy_score(y_test, nb_pred):.4f}")
    st.text(classification_report(y_test, nb_pred))
    st.write(confusion_matrix(y_test, nb_pred))

    st.subheader("SVM Results")
    st.write(f"Accuracy: {accuracy_score(y_test, svm_pred):.4f}")
    st.text(classification_report(y_test, svm_pred))
    st.write(confusion_matrix(y_test, svm_pred))

    # Message prediction UI
    st.subheader("🔮 Try Predicting Your Own Message")
    user_msg = st.text_area("Enter a message to test:")
    model_choice = st.selectbox("Choose a model", ["Naive Bayes", "SVM"])

    if st.button("Predict"):
        if user_msg.strip():
            processed = preprocess_text(user_msg)
            if model_choice == "Naive Bayes":
                pred = nb_pipeline.predict([processed])[0]
                prob = nb_pipeline.predict_proba([processed])[0][pred]
            else:
                pred = svm_pipeline.predict([processed])[0]
                prob = svm_pipeline.predict_proba([processed])[0][pred]
            label = "Spam" if pred == 1 else "Not Spam"
            st.success(f"Prediction: **{label}** with confidence {prob * 100:.2f}%")
        else:
            st.warning("Please enter a message.")
else:
    st.info("👈 Upload a dataset to begin training the models.")
