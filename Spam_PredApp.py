# app.py
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import make_pipeline

@st.cache_data
def load_data(path: str):
    df = pd.read_csv(path)
    df = df[['label', 'message']].dropna()
    df['label_num'] = df['label'].map({'ham': 0, 'spam': 1})
    return df

@st.cache_resource
def train_model(df: pd.DataFrame):
    X_train, X_test, y_train, y_test = train_test_split(
        df['message'], df['label_num'],
        test_size=0.2, random_state=42, stratify=df['label_num']
    )

    # TF–IDF with bigrams + Complement Naive Bayes
    pipeline = make_pipeline(
        TfidfVectorizer(stop_words='english', ngram_range=(1, 2)),
        ComplementNB(alpha=0.1)
    )
    pipeline.fit(X_train, y_train)

    # (Optional) Evaluate once on held‑out test set
    # from sklearn.metrics import classification_report
    # print(classification_report(y_test, pipeline.predict(X_test),
    #       target_names=['ham','spam']))

    return pipeline

def main():
    st.set_page_config(page_title="SMS Spam Classifier", layout="centered")
    st.title("📨 SMS Spam Classifier (Improved)")
    st.write(
        "Type your message below and click **Predict**. "
        "This version uses n‑grams and ComplementNB for better spam recall."
    )

    df = load_data("SMSSpamCollection.csv")
    model = train_model(df)

    user_input = st.text_area("Your message:", height=150)
    if st.button("Predict"):
        if not user_input.strip():
            st.warning("Please enter some text to classify.")
        else:
            pred = model.predict([user_input])[0]
            label = "🟢 Ham (Not Spam)" if pred == 0 else "🔴 Spam"
            st.success(f"Prediction: **{label}**")

    with st.expander("📊 Dataset overview"):
        st.write(f"Total messages: {len(df)}")
        st.write(df['label'].value_counts())

if __name__ == "__main__":
    main()
