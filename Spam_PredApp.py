# Import required libraries
import streamlit as st
import pandas as pd
import nltk
import string

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import make_pipeline

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Enhanced preprocessing function
def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    
    # Remove punctuation
    text_no_punct = text.translate(str.maketrans('', '', string.punctuation))
    tokens = nltk.word_tokenize(text_no_punct)
    stop_words = set(nltk.corpus.stopwords.words('english'))
    ps = nltk.PorterStemmer()
    processed_tokens = [ps.stem(word.lower()) for word in tokens if word.lower() not in stop_words]
    processed_text = ' '.join(processed_tokens)
    
    # Fallback: if processed text is empty, return the original text in lower case
    return processed_text if processed_text.strip() != "" else text.lower()

# Streamlit app configuration
st.set_page_config(page_title="Spam Predictor", layout="wide")
st.title("📧 Spam Message Classifier")
st.markdown("""
Upload a dataset and predict if messages are **Spam** or **Not Spam** using ML models!
""")

# Sidebar: File upload and settings
st.sidebar.header("Upload & Settings")
upload_file = st.sidebar.file_uploader("Upload CSV Data (columns: label, message)", type=["csv", "txt"])
test_size = st.sidebar.slider("Test Set Size (%)", 10, 40, 20)

if upload_file:
    try:
        # Try reading as a tab-separated file (like SMSSpamCollection)
        df = pd.read_csv(upload_file, sep='\t', header=None, names=['label', 'message'])
    except Exception:
        try:
            # Fallback: try reading as comma-separated
            df = pd.read_csv(upload_file)
        except Exception as e:
            st.error(f"Could not read file: {e}")
            st.stop()

    # Verify that required columns exist
    if 'label' not in df.columns or 'message' not in df.columns:
        st.error("CSV must contain 'label' and 'message' columns.")
        st.stop()

    # Drop rows with null labels or messages
    df.dropna(subset=['label', 'message'], inplace=True)

    # Map labels: convert them to lowercase and then to 0/1
    df['label'] = df['label'].astype(str).str.lower().map({'ham': 0, 'spam': 1})
    df.dropna(subset=['label'], inplace=True)  # Remove rows with labels that did not match

    st.write("### Data Sample")
    st.dataframe(df.head())

    st.write("Preprocessing messages...")
    df['processed_text'] = df['message'].apply(preprocess_text)
    
    # Filter out rows where processed_text is empty (as a last check)
    df = df[df['processed_text'].apply(lambda x: isinstance(x, str) and x.strip() != "")]
    
    # Check if sufficient data is available for training
    if len(df) < 5:
        st.error("Insufficient valid data after preprocessing. Please upload a larger or cleaner dataset.")
        st.stop()

    # Prepare features and labels
    X = df['processed_text']
    y = df['label']

    # Train-test split with stratification
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size/100, stratify=y, random_state=42
        )
    except ValueError as e:
        st.error(f"Train-test split failed: {e}")
        st.stop()

    # Build and train the Naive Bayes model pipeline
    nb_pipeline = make_pipeline(TfidfVectorizer(), MultinomialNB())
    nb_pipeline.fit(X_train, y_train)
    nb_pred = nb_pipeline.predict(X_test)

    # Build and train the SVM model pipeline
    svm_pipeline = make_pipeline(TfidfVectorizer(), SVC(kernel='linear', probability=True))
    svm_pipeline.fit(X_train, y_train)
    svm_pred = svm_pipeline.predict(X_test)

    # Display Naive Bayes results
    st.markdown("### Naive Bayes Model Results")
    st.write(f"**Accuracy:** {accuracy_score(y_test, nb_pred):.4f}")
    st.text(classification_report(y_test, nb_pred))
    st.write("**Confusion Matrix:**")
    st.write(confusion_matrix(y_test, nb_pred))

    # Display SVM results
    st.markdown("### SVM Model Results")
    st.write(f"**Accuracy:** {accuracy_score(y_test, svm_pred):.4f}")
    st.text(classification_report(y_test, svm_pred))
    st.write("**Confusion Matrix:**")
    st.write(confusion_matrix(y_test, svm_pred))

    # Prediction interface for individual messages
    st.markdown("### 🔮 Test a Message")
    test_message = st.text_area("Enter your message:")
    model_choice = st.selectbox("Choose a model", ["Naive Bayes", "SVM"])

    if st.button("Predict"):
        if test_message.strip() == "":
            st.warning("Please enter a message to predict.")
        else:
            processed_msg = preprocess_text(test_message)
            if model_choice == "Naive Bayes":
                pred = nb_pipeline.predict([processed_msg])[0]
                prob = nb_pipeline.predict_proba([processed_msg])[0][pred]
            else:
                pred = svm_pipeline.predict([processed_msg])[0]
                prob = svm_pipeline.predict_proba([processed_msg])[0][pred]
            label = "Spam" if pred == 1 else "Not Spam"
            st.success(f"Prediction: **{label}** (Confidence: {prob * 100:.1f}%)")
else:
    st.info("👈 Please upload a dataset to begin.")
