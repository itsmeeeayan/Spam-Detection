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

# Text preprocessing function
def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = nltk.word_tokenize(text)
    stop_words = set(nltk.corpus.stopwords.words('english'))
    ps = nltk.PorterStemmer()
    processed = [ps.stem(word.lower()) for word in tokens if word.lower() not in stop_words]
    return ' '.join(processed) if processed else text

# Streamlit app configuration
st.set_page_config(page_title="Spam Predictor", layout="wide")
st.title("📧 Spam Message Classifier")
st.markdown("""
Upload a dataset and predict if messages are **Spam** or **Not Spam** using ML models!
""")

# Sidebar
st.sidebar.header("Upload & Settings")
upload_file = st.sidebar.file_uploader("Upload a CSV File", type=["csv", "txt"])
test_size = st.sidebar.slider("Test Set Size (%)", 10, 40, 20)

# Main logic
if upload_file:
    try:
        df = pd.read_csv(upload_file, sep='\t', header=None, names=['label', 'message'])
    except Exception:
        try:
            df = pd.read_csv(upload_file)
        except Exception as e:
            st.error(f"Could not read file: {e}")
            st.stop()

    # Basic column check
    if 'label' not in df.columns or 'message' not in df.columns:
        st.error("CSV must contain 'label' and 'message' columns.")
        st.stop()

    # Drop nulls
    df.dropna(subset=['label', 'message'], inplace=True)

    # Map labels
    df['label'] = df['label'].astype(str).str.lower().map({'ham': 0, 'spam': 1})
    df.dropna(subset=['label'], inplace=True)

    # Preprocess text
    st.write("Preprocessing text...")
    df['processed_text'] = df['message'].apply(preprocess_text)

    # Filter valid processed text
    df = df[df['processed_text'].apply(lambda x: isinstance(x, str) and x.strip() != "")]
    if 'processed_text' not in df.columns or len(df) < 5:
        st.error("Insufficient valid data after preprocessing. Please upload a larger or cleaner dataset.")
        st.stop()

    st.write("### Sample Data")
    st.dataframe(df.head())

    # Features and labels
    X = df['processed_text']
    y = df['label']

    # Train-test split
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size/100, stratify=y, random_state=42
        )
    except ValueError as e:
        st.error(f"Train-test split failed: {e}")
        st.stop()

    # Train models
    nb_model = make_pipeline(TfidfVectorizer(), MultinomialNB())
    nb_model.fit(X_train, y_train)
    nb_pred = nb_model.predict(X_test)

    svm_model = make_pipeline(TfidfVectorizer(), SVC(kernel='linear', probability=True))
    svm_model.fit(X_train, y_train)
    svm_pred = svm_model.predict(X_test)

    # Results
    st.subheader("Naive Bayes Results")
    st.write(f"Accuracy: **{accuracy_score(y_test, nb_pred):.4f}**")
    st.text(classification_report(y_test, nb_pred))
    st.write("Confusion Matrix:", confusion_matrix(y_test, nb_pred))

    st.subheader("SVM Results")
    st.write(f"Accuracy: **{accuracy_score(y_test, svm_pred):.4f}**")
    st.text(classification_report(y_test, svm_pred))
    st.write("Confusion Matrix:", confusion_matrix(y_test, svm_pred))

    # Prediction input
    st.subheader("🔮 Test a Message")
    test_message = st.text_area("Enter message:")
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
            st.success(f"Prediction: **{label}** ({prob*100:.1f}% confidence)")
else:
    st.info("👈 Please upload a dataset to begin.")
