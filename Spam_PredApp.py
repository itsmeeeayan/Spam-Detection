# Import necessary libraries
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

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Preprocessing function for text, with a fallback for empty results.
def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    # Remove punctuation
    processed = text.translate(str.maketrans('', '', string.punctuation))
    tokens = nltk.word_tokenize(processed)
    stop_words = set(nltk.corpus.stopwords.words('english'))
    ps = nltk.PorterStemmer()
    processed_tokens = [ps.stem(word.lower()) for word in tokens if word.lower() not in stop_words]
    processed_text = ' '.join(processed_tokens)
    # Fallback to original text if processed text is empty
    return processed_text if processed_text.strip() != "" else text

# Streamlit page configuration
st.set_page_config(page_title="Spam Prediction App", layout="wide")
st.title("📧 Spam Prediction Web App")
st.markdown("""
This app predicts whether a message is **spam** or **not spam** using ML algorithms (Naive Bayes & SVM).
""")

# Sidebar for upload and settings
st.sidebar.header("Data Upload & Settings")
upload_file = st.sidebar.file_uploader("Upload CSV Data (columns: label, message)", type=["csv"])
test_size = st.sidebar.slider("Test Set Size (%)", 10, 40, 20)

if upload_file:
    try:
        # Read CSV file; for SMSSpamCollection, assuming tab-separated without header
        df = pd.read_csv(upload_file, sep='\t', header=None, names=['label', 'message'])
    except Exception:
        # Alternatively, if CSV is comma-separated
        df = pd.read_csv(upload_file)

    # Check if the CSV contains the expected columns
    if 'label' not in df.columns or 'message' not in df.columns:
        st.error("CSV must have 'label' and 'message' columns.")
        st.stop()

    df.dropna(subset=['label', 'message'], inplace=True)

    # Ensure label consistency: convert to lowercase for mapping
    df['label'] = df['label'].astype(str).str.lower().map({'ham': 0, 'spam': 1})
    # Drop any rows where label conversion failed
    df.dropna(subset=['label'], inplace=True)
    
    st.write("### Data Sample")
    st.write(df.head())

    st.write("Preprocessing messages...")
    df['processed_text'] = df['message'].apply(preprocess_text)
    # Remove rows with empty processed text
    df = df[df['processed_text'].str.strip() != ""]
    
    # Check if data is sufficient
    if len(df) < 5:
        st.error("Insufficient data after preprocessing. Upload a larger dataset.")
        st.stop()
    
    # Prepare features and labels
    X = df['processed_text']
    y = df['label']
    
    # Train-test split (using stratify to maintain label distribution)
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size / 100, stratify=y, random_state=42
        )
    except ValueError as e:
        st.error(f"Train-test split failed: {e}")
        st.stop()
    
    # ---------------------------
    # Build and Train Pipelines
    # ---------------------------
    nb_pipeline = make_pipeline(TfidfVectorizer(), MultinomialNB())
    nb_pipeline.fit(X_train, y_train)
    nb_pred = nb_pipeline.predict(X_test)
    
    svm_pipeline = make_pipeline(TfidfVectorizer(), SVC(kernel='linear', probability=True))
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
            st.success(f"Prediction: **{label}** (Confidence: {prob * 100:.1f}%)")
else:
    st.info("👈 Please upload a CSV file to get started with the spam prediction model.")
