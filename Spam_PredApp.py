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

# Updated text preprocessing function:
# If processed text is empty, revert to the original text.
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
    # Fallback to original text if the processed result is empty
    return processed_text if processed_text.strip() != "" else text

# Streamlit app configuration
st.set_page_config(page_title="Spam Predictor", layout="wide")
st.title("📧 Spam Message Classifier")
st.markdown("""
Upload a dataset and predict if messages are **Spam** or **Not Spam** using ML models!
""")

# Sidebar - Upload and Settings
st.sidebar.header("Upload & Settings")
upload_file = st.sidebar.file_uploader("Upload a CSV File", type=["csv", "txt"])
test_size = st.sidebar.slider("Test Set Size (%)", 10, 40, 20)

if upload_file:
    try:
        # Try reading as tab-separated CSV (SMSSpamCollection format)
        df = pd.read_csv(upload_file, sep='\t', header=None, names=['label', 'message'])
    except Exception:
        # Fallback: try reading as comma-separated file
        try:
            df = pd.read_csv(upload_file)
        except Exception as e:
            st.error(f"Could not read file: {e}")
            st.stop()

    # Check if the required columns exist
    if 'label' not in df.columns or 'message' not in df.columns:
        st.error("CSV must contain 'label' and 'message' columns.")
        st.stop()

    # Drop nulls
    df.dropna(subset=['label', 'message'], inplace=True)

    # Map labels: convert to lowercase and map ham/spam to 0/1
    df['label'] = df['label'].astype(str).str.lower().map({'ham': 0, 'spam': 1})
    df.dropna(subset=['label'], inplace=True)

    st.write("### Sample Data")
    st.dataframe(df.head())

    st.write("Preprocessing messages...")
    df['processed_text'] = df['message'].apply(preprocess_text)

    # Instead of filtering out too many rows, we check if the processed text is valid.
    # We don't filter out messages that are non-empty, as our updated function now falls back to original text.
    df = df[df['processed_text'].apply(lambda x: isinstance(x, str) and x.strip() != "")]
    
    # Check if we have sufficient data (update threshold if needed)
    if len(df) < 5:
        st.error("Insufficient valid data after preprocessing. Please upload a larger or cleaner dataset.")
        st.stop()

    # Prepare features and labels
    X = df['processed_text']
    y = df['label']

    # Train-test split with stratification to maintain label balance
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size/100, stratify=y, random_state=42
        )
    except ValueError as e:
        st.error(f"Train-test split failed: {e}")
        st.stop()

    # Build and train Naive Bayes pipeline
    nb_model = make_pipeline(TfidfVectorizer(), MultinomialNB())
    nb_model.fit(X_train, y_train)
    nb_pred = nb_model.predict(X_test)

    # Build and train SVM pipeline
    svm_model = make_pipeline(TfidfVectorizer(), SVC(kernel='linear', probability=True))
    svm_model.fit(X_train, y_train)
    svm_pred = svm_model.predict(X_test)

    # Display Naive Bayes Results
    st.subheader("Naive Bayes Model Results")
    st.write(f"Accuracy: **{accuracy_score(y_test, nb_pred):.4f}**")
    st.text(classification_report(y_test, nb_pred))
    st.write("Confusion Matrix:", confusion_matrix(y_test, nb_pred))

    # Display SVM Results
    st.subheader("SVM Model Results")
    st.write(f"Accuracy: **{accuracy_score(y_test, svm_pred):.4f}**")
    st.text(classification_report(y_test, svm_pred))
    st.write("Confusion Matrix:", confusion_matrix(y_test, svm_pred))

    # Prediction Interface
    st.subheader("🔮 Test a Message for Spam/Not Spam")
    test_message = st.text_area("Enter your message:")
    model_choice = st.selectbox("Choose a model", ["Naive Bayes", "SVM"])

    if st.button("Predict"):
        if test_message.strip() == "":
            st.warning("Please enter a message to predict.")
        else:
            processed_msg = preprocess_text(test_message)
            model = nb_model if model_choice == "Naive Bayes" else svm_model
            prediction = model.predict([processed_msg])[0]
            prob = model.predict_proba([processed_msg])[0][prediction]
            label = "Spam" if prediction == 1 else "Not Spam"
            st.success(f"Prediction: **{label}** ({prob * 100:.1f}% confidence)")
else:
    st.info("👈 Please upload a dataset to get started with the spam prediction model.")
