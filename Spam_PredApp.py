# app.py
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import make_pipeline

# 1) Apply the background image
def add_background():
    st.markdown(
        """
        <style>
        .stApp {
            background-image: url("https://images.unsplash.com/photo-1581090700227-c391f23a3bab");
            background-size: cover;
            background-position: center;
        }
        .css-18e3th9 {  /* Main content area transparency */
            background-color: rgba(44, 62, 80, 0.6);
        }
        .css-1d391kg {  /* Sidebar transparency */
            background-color: rgba(52, 73, 94, 0.8);
        }
        </style>
        """,
        unsafe_allow_html=True
    )

# 2) Load and preprocess data
@st.cache_data
def load_data(path: str):
    df = pd.read_csv(path)
    df = df[['label', 'message']].dropna()
    df['label_num'] = df['label'].map({'ham': 0, 'spam': 1})
    return df

# 3) Train ComplementNB with TF–IDF (1–2 grams)
@st.cache_resource
def train_model(df: pd.DataFrame):
    X_train, X_test, y_train, y_test = train_test_split(
        df['message'], df['label_num'],
        test_size=0.2, random_state=42, stratify=df['label_num']
    )
    pipeline = make_pipeline(
        TfidfVectorizer(stop_words='english', ngram_range=(1, 2)),
        ComplementNB(alpha=0.1)
    )
    pipeline.fit(X_train, y_train)
    return pipeline

def main():
    st.add_background()
    # 4) Page config with icon and centered layout
    st.set_page_config(
        page_title="📨 SMS Spam Classifier",
        page_icon="✉️",
        layout="centered"
    )

    # 5) Inject background and theme
    add_background()

    st.title("📨 SMS Spam Classifier")
    st.markdown(
        "Enter your message below and click **Predict** to see if it’s **Spam** or **Ham**.",
    )

    df = load_data("SMSSpamCollection.csv")
    model = train_model(df)

    # 6) Interactive UI
    user_input = st.text_area("Type your message here:", height=150)
    if st.button("Predict"):
        if not user_input.strip():
            st.warning("Please enter some text before predicting.")
        else:
            pred = model.predict([user_input])[0]
            label = "🟢 Ham (Not Spam)" if pred == 0 else "🔴 Spam"
            st.success(f"**Prediction:** {label}")

    # 7) Sidebar stats (themed automatically)
    with st.sidebar.expander("📊 Dataset Overview"):
        st.write(f"Total messages: {len(df)}")
        st.write(df['label'].value_counts())

if __name__ == "__main__":
    main()
