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
# If the resulting processed text is empty, fall back to the original message.
def preprocess_text(text):
    if not isinstance(text, str) or text.strip() == "":
        return text
    # Remove punctuation
    processed = text.translate(str.maketrans('', '', string.punctuation))
    tokens = nltk.word_tokenize(processed)
    stop_words = set(nltk.corpus.stopwords.words('english'))
    ps = nltk.PorterStemmer()
    processed_tokens = [ps.stem(word.lower()) for word in tokens if word.lower() not in stop_words]
    processed_text = ' '.join(processed_tokens)
    # If result is empty, fallback to the original text
    return processed_text if processed_text.strip() != "" else text

# Streamlit app configuration
st.set_page_config(page_title="Spam Predictor", layout="wide")
st.title("📧 Spam Message Classifier")
st.markdown("Upload a dataset and predict if messages are **Spam** or **Not Spam** using ML models!")

# Sidebar - Upload and Settings
st.sidebar.header("Upload & Settings")
upload_file = st.sidebar.file_uploader("Upload a CSV File", type=["csv", "txt"])
test_size = st.sidebar.slider("Test Set Size (%)", 10, 40, 20)

if upload_file:
    # Attempt to read file using tab-separated format first (SMSSpamCollection format)
    try:
        df = pd.read_csv(upload_file, sep='\t', header=None, names=['label', 'message'])
    except Exception:
        try:
            df = pd.read_csv(upload_file)
        except Exception as e:
            st.error(f"Could not read file: {e}")
            st.stop()
    
    # Verify that required columns exist
    if 'label' not in df.columns or 'message' not in df.columns:
        st.error("CSV must contain 'label' and 'message' columns.")
        st.stop()
    
    # Drop any rows with missing values in required columns
    df.dropna(subset=['label', 'message'], inplace=True)
    
    # Map labels: convert to lowercase and map 'ham' to 0 and 'spam' to 1
    df['label'] = df['label'].astype(str).str.lower().map({'ham': 0, 'spam': 1})
    df.dropna(subset=['label'], inplace=True)
    
    st.write("### Sample Data")
    st.dataframe(df.head())

    st.write("Preprocessing messages...")
    # Create a new column with processed text
    df['processed_text'] = df['message'].apply(preprocess_text)
    
    # Debug info: show number of rows before filtering
    st.write("Rows before filtering:", df.shape[0])
    
    # Filter out rows where processed_text is empty
    filtered_df = df[df['processed_text'].apply(lambda x: isinstance(x, str) and x.strip() != "")]
    st.write("Rows after filtering:", filtered_df.shape[0])
    
    # If after filtering there are fewer than 5 rows, use original messages instead.
    if filtered_df.shape[0] < 5:
        st.warning("Insufficient valid processed data. Using original messages for training.")
        df['processed_text'] = df['message']
    else:
        df = filtered_df

    st.write("Rows after final processing:", df.shape[0])
    
    # Prepare features and labels
    X = df['processed_text']
    y = df['label']
    
    # Perform train-test split using stratification to preserve label balance
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size / 100, stratify=y, random_state=42
        )
    except ValueError as e:
        st.error(f"Train-test split failed: {e}")
        st.stop()
    
    # Build and train the Naive Bayes pipeline
    nb_model = make_pipeline(TfidfVectorizer(), MultinomialNB())
    nb_model.fit(X_train, y_train)
    nb_pred = nb_model.predict(X_test)
    
    # Build and train the SVM pipeline
    svm_model = make_pipeline(TfidfVectorizer(), SVC(kernel='linear', probability=True))
    svm_model.fit(X_train, y_train)
    svm_pred = svm_model.predict(X_test)
    
    # Display results for Naive Bayes
    st.subheader("Naive Bayes Model Results")
    st.write(f"Accuracy: **{accuracy_score(y_test, nb_pred):.4f}**")
    st.text(classification_report(y_test, nb_pred))
    st.write("Confusion Matrix:", confusion_matrix(y_test, nb_pred))
    
    # Display results for SVM
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
