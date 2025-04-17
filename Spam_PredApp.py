# app.py
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

@st.cache_data(show_spinner=False)
def load_data(path: str):
    # Load dataset; adjust delimiter if needed
    df = pd.read_csv(path)
    # Ensure only the two expected columns exist
    df = df[['label', 'message']].dropna()
    # Map labels to binary
    df['label_num'] = df['label'].map({'ham': 0, 'spam': 1})
    return df

@st.cache_resource(show_spinner=False)
def train_model(df: pd.DataFrame):
    X_train, X_test, y_train, y_test = train_test_split(
        df['message'], df['label_num'], test_size=0.2, random_state=42
    )
    # Build a pipeline: TF–IDF vectorizer + MultinomialNB
    model = make_pipeline(
        TfidfVectorizer(stop_words='english'),
        MultinomialNB()
    )
    model.fit(X_train, y_train)
    return model

def main():
    st.set_page_config(page_title="SMS Spam Classifier", layout="centered")
    st.title("📨 SMS Spam Classifier")
    st.write(
        "Enter a message below and click **Predict** to see whether it's **Spam** or **Ham**."
    )

    # Load data & train
    df = load_data("Spam.csv")
    model = train_model(df)

    # User input
    user_input = st.text_area("Your message:", height=150)
    if st.button("Predict"):
        if not user_input.strip():
            st.warning("Please enter some text to classify.")
        else:
            pred = model.predict([user_input])[0]
            label = "🟢 Ham (Not Spam)" if pred == 0 else "🔴 Spam"
            st.success(f"Prediction: **{label}**")

    # (Optional) show dataset stats in an expander
    with st.expander("📊 Dataset overview"):
        st.write(f"Total messages: {len(df)}")
        st.write(df['label'].value_counts())

if __name__ == "__main__":
    main()
