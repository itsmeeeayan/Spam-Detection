# Import necessary libraries
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

# Preprocessing function with added check for string input
def preprocess_text(text):
    # If the text is NaN or not a string, convert or return empty string
    if not isinstance(text, str):
        text = str(text) if pd.notnull(text) else ""
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Tokenize text
    tokens = nltk.word_tokenize(text)
    # Remove stopwords and apply stemming
    stop_words = set(nltk.corpus.stopwords.words('english'))
    ps = nltk.PorterStemmer()
    processed_tokens = [ps.stem(word.lower()) for word in tokens if word.lower() not in stop_words]
    return ' '.join(processed_tokens)

# ------------------------------
# Set page configuration for Streamlit
# ------------------------------
st.set_page_config(
    page_title="Kolkata Weather Forecast",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Optionally, set a background image using CSS (update URL as needed)
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

def main():
    st.title("📧 Spam Prediction Web App")
    st.markdown("""
    This interactive app allows you to build a spam prediction model using Machine Learning (Naive Bayes and SVM) and check if your messages are spam or not.
    
    **Steps:**
    1. Upload a CSV dataset (with 'label' and 'message' columns).
    2. The dataset is preprocessed, and two models are trained:
       - Naive Bayes (MultinomialNB)
       - Support Vector Machine (SVM with linear kernel)
    3. Evaluation metrics and confusion matrices are displayed.
    4. You can then input your own message to see whether it is spam.
    """)

    st.sidebar.header("Model Settings & Data Upload")
    upload_file = st.sidebar.file_uploader("Upload CSV Data", type=["csv"])
    test_size = st.sidebar.slider("Test Set Size (%)", 10, 40, 20)

    if upload_file is not None:
        try:
            # Load dataset; expected format: two columns without headers.
            # For instance: the SMSSpamCollection dataset with tab-separated values.
            df = pd.read_csv(upload_file, sep='\t', header=None, names=['label', 'message'])
        except Exception as e:
            st.error(f"Error reading file: {e}")
            st.stop()
    
        # Map labels: 'ham' -> 0, 'spam' -> 1.
        df['label'] = df['label'].map({'ham': 0, 'spam': 1})
    
        st.write("### Original Data")
        st.write(df.head())
    
        # Preprocess messages with error-handling function.
        st.write("Preprocessing messages...")
        df['processed_text'] = df['message'].apply(preprocess_text)
        st.write("### Preprocessed Data")
        st.write(df.head())
    
        # Split the dataset
        X = df['processed_text']
        y = df['label']
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size/100, random_state=42, stratify=y
        )
    
        # ---------------------------
        # Model Pipelines Setup
        # ---------------------------
        nb_pipeline = make_pipeline(
            TfidfVectorizer(),
            MultinomialNB()
        )
        svm_pipeline = make_pipeline(
            TfidfVectorizer(),
            SVC(kernel='linear', probability=True)
        )
    
        # Train the models
        st.markdown("### Training Models...")
        nb_pipeline.fit(X_train, y_train)
        svm_pipeline.fit(X_train, y_train)
    
        nb_pred = nb_pipeline.predict(X_test)
        svm_pred = svm_pipeline.predict(X_test)
    
        # Evaluate models
        st.markdown("### Naive Bayes Model Results")
        st.write(f"**Accuracy:** {accuracy_score(y_test, nb_pred):.4f}")
        st.text("Classification Report:")
        st.text(classification_report(y_test, nb_pred))
        st.write("Confusion Matrix:")
        st.write(confusion_matrix(y_test, nb_pred))
    
        st.markdown("### SVM Model Results")
        st.write(f"**Accuracy:** {accuracy_score(y_test, svm_pred):.4f}")
        st.text("Classification Report:")
        st.text(classification_report(y_test, svm_pred))
        st.write("Confusion Matrix:")
        st.write(confusion_matrix(y_test, svm_pred))
    
        # ---------------------------
        # Prediction Interface
        # ---------------------------
        st.markdown("### 🔮 Test a Message for Spam/Not Spam")
        user_message = st.text_area("Enter your message below:")
        model_choice = st.selectbox("Choose a Model for Prediction", ["Naive Bayes", "SVM"])
    
        if st.button("Predict Message"):
            if user_message.strip() != "":
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
        st.info("👈 Please upload a CSV file to get started with the spam prediction model.")

if __name__ == "__main__":
    main()
