import streamlit as st
import pandas as pd
import numpy as np
import nltk
import string
import matplotlib.pyplot as plt
import time

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

# Set page configuration BEFORE any other streamlit commands
st.set_page_config(
    page_title="Kolkata Weather Forecast",
    layout="wide",
    initial_sidebar_state="expanded"
)

# (Optional) Set a background image, etc.
def set_bg_image():
    st.markdown(
        """
        <style>
        .stApp {
            background: url("https://raw.githubusercontent.com/username/repository/main/path/to/image.jpg");
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

# Continue with the rest of your app code...
def load_and_preprocess_data(file):
    try:
        df = pd.read_csv(file)
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None

    df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y')
    df['day_of_year'] = df['Date'].dt.dayofyear
    df['month'] = df['Date'].dt.month

    if 'preciptype' in df.columns:
        df['preciptype'] = df['preciptype'].fillna('none')
    else:
        df['preciptype'] = 'none'

    df['precipitation_flag'] = np.where(df['precip'] > 0, 1, 0)
    return df

def main():
    set_bg_image()
    st.title("🌤️ Kolkata Weather Forecasting Web App")
    st.markdown("""
    This interactive app uses machine learning to predict temperatures in Kolkata.
    Two models are trained on historical weather data:
    - **Linear Regression**
    - **Decision Tree Regressor**
    
    You can view the model performances, inspect interactive plots, and even predict temperature 
    for your own set of weather parameters.
    """)

    st.sidebar.header("Upload Data")
    upload_file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])
    
    if upload_file is not None:
        df = load_and_preprocess_data(upload_file)
        if df is not None:
            st.subheader("Data Overview")
            st.write(df.head())

            test_size = st.sidebar.slider("Test Set Size (%)", 10, 40, 20)
            max_depth = st.sidebar.slider("Decision Tree Max Depth", 2, 10, 5)
            
            features = [
                'day_of_year', 'humidity', 'sealevelpressure', 'precip',
                'windspeed', 'cloudcover', 'month', 'precipitation_flag'
            ]
            target = 'temp'
            
            preprocessor = ColumnTransformer(
                transformers=[
                    ('cat', OneHotEncoder(), ['conditions'])
                ],
                remainder='passthrough'
            )
            
            X = df[features + ['conditions']]
            y = df[target]
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size/100, random_state=42, stratify=y
            )
            
            X_train_processed = preprocessor.fit_transform(X_train)
            X_test_processed = preprocessor.transform(X_test)
            
            st.markdown("### Linear Regression Model Training")
            lr_model = LinearRegression()
            lr_model.fit(X_train_processed, y_train)
            lr_predictions = lr_model.predict(X_test_processed)
            
            lr_mae = mean_absolute_error(y_test, lr_predictions)
            lr_mse = mean_squared_error(y_test, lr_predictions)
            lr_r2  = r2_score(y_test, lr_predictions)
            
            st.write("**Linear Regression Metrics:**")
            st.write(f"MAE: {lr_mae:.2f}°C")
            st.write(f"MSE: {lr_mse:.2f}")
            st.write(f"R² Score: {lr_r2:.2f}")
            
            st.markdown("### Decision Tree Model Training")
            dt_model = DecisionTreeRegressor(max_depth=max_depth, random_state=42)
            dt_model.fit(X_train_processed, y_train)
            dt_predictions = dt_model.predict(X_test_processed)
            
            dt_mae = mean_absolute_error(y_test, dt_predictions)
            dt_mse = mean_squared_error(y_test, dt_predictions)
            dt_r2  = r2_score(y_test, dt_predictions)
            
            st.write("**Decision Tree Metrics:**")
            st.write(f"MAE: {dt_mae:.2f}°C")
            st.write(f"MSE: {dt_mse:.2f}")
            st.write(f"R² Score: {dt_r2:.2f}")
            
            st.markdown("### Predictions Visualization")
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
            ax1.plot(y_test.values, label='Actual', marker='o')
            ax1.plot(lr_predictions, label='Linear Regression', marker='x')
            ax1.plot(dt_predictions, label='Decision Tree', marker='s')
            ax1.set_xlabel('Data Points')
            ax1.set_ylabel('Temperature (°C)')
            ax1.set_title('Actual vs Predicted Temperatures')
            ax1.legend()
    
            ax2.scatter(lr_predictions, lr_predictions - y_test, alpha=0.5, label='Linear Regression')
            ax2.scatter(dt_predictions, dt_predictions - y_test, alpha=0.5, color='green', label='Decision Tree')
            ax2.axhline(y=0, color='r', linestyle='--')
            ax2.set_xlabel('Predicted Values')
            ax2.set_ylabel('Residuals')
            ax2.set_title('Residual Analysis')
            ax2.legend()
            
            st.pyplot(fig, use_container_width=True)
            
            st.markdown("### Decision Tree Feature Importance")
            cat_feature_names = preprocessor.named_transformers_['cat'].get_feature_names_out(['conditions']).tolist()
            feature_names = cat_feature_names + features
            importances = dt_model.feature_importances_
            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': importances
            }).sort_values('Importance', ascending=False)
            st.bar_chart(importance_df.set_index('Feature'))
            
            st.markdown("### 🔮 Make a Temperature Prediction")
            with st.form("prediction_form"):
                st.write("Enter weather parameters:")
                col1, col2, col3 = st.columns(3)
                with col1:
                    date = st.date_input("Date")
                    humidity_input = st.number_input("Humidity (%)", 0, 100, 60)
                with col2:
                    pressure_input = st.number_input("Sea Level Pressure (hPa)", 980, 1040, 1010)
                    precip_input = st.number_input("Precipitation (mm)", 0.0, 50.0, 0.0)
                with col3:
                    wind_speed_input = st.number_input("Wind Speed (km/h)", 0.0, 50.0, 10.0)
                    cloud_cover_input = st.number_input("Cloud Cover (%)", 0, 100, 30)
    
                condition = st.selectbox("Weather Condition", df['conditions'].unique())
    
                submit_button = st.form_submit_button("Predict Temperature")
    
            if submit_button:
                input_data = pd.DataFrame([{
                    'day_of_year': pd.Timestamp(date).dayofyear,
                    'month': pd.Timestamp(date).month,
                    'humidity': humidity_input,
                    'sealevelpressure': pressure_input,
                    'precip': precip_input,
                    'windspeed': wind_speed_input,
                    'cloudcover': cloud_cover_input,
                    'precipitation_flag': 1 if precip_input > 0 else 0,
                    'conditions': condition
                }])
                processed_input = preprocessor.transform(input_data)
                lr_pred_value = lr_model.predict(processed_input)[0]
                dt_pred_value = dt_model.predict(processed_input)[0]
    
                st.markdown("#### Prediction Results")
                col_lr, col_dt = st.columns(2)
                with col_lr:
                    st.metric("Linear Regression Prediction", f"{lr_pred_value:.1f}°C")
                with col_dt:
                    st.metric("Decision Tree Prediction", f"{dt_pred_value:.1f}°C")
    
    else:
        st.info("👈 Please upload a CSV file to get started with the weather forecast data.")
    
if __name__ == "__main__":
    main()
