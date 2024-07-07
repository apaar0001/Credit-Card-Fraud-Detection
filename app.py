import streamlit as st
import pandas as pd
import joblib
from PIL import Image
from Model import preprocess_data, apply_pca  # Adjust import path as per your project structure

# Load trained models
@st.cache_resource
def load_models():
    models = {
        'RandomForest': joblib.load('Models/RandomForest_model.joblib'),
        'GradientBoosting': joblib.load('Models/GradientBoosting_model.joblib'),
        'LogisticRegression': joblib.load('Models/LogisticRegression_model.joblib'),
        'NearestCentroid': joblib.load('Models/NearestCentroid_model.joblib')
    }
    return models

# Load classification reports
def load_classification_report(model_name):
    try:
        file_path = f'Results/classification_report_{model_name}.txt'
        with open(file_path, 'r') as f:
            report = f.read()
        return report
    except FileNotFoundError:
        return f"Classification report for {model_name} not found."

# Load dataset
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('Datasets/creditcard.csv')
        return df
    except FileNotFoundError:
        st.error("Dataset not found. Please ensure the dataset is placed in the 'Datasets' directory.")
        return None

# Main function to run the app
def main():
    st.title('Credit Card Fraud Detection')

    # Load data
    df = load_data()
    if df is None:
        return

    # Load models
    models = load_models()

    # Sidebar for user inputs
    st.sidebar.subheader('Model Selection')
    model_choice = st.sidebar.selectbox('Choose a Model', list(models.keys()))

    if st.sidebar.checkbox('Show Data Summary'):
        st.subheader('Data Summary')
        st.write(df.head())

    if st.sidebar.checkbox('Show Class Distribution'):
        st.subheader('Class Distribution')
        class_counts = df['Class'].value_counts()
        st.write(class_counts)

    # Preprocess data
    df_processed = preprocess_data(df.copy())

    # Feature engineering (PCA - Optional)
    X = df_processed.drop('Class', axis=1)
    _, X_pca = apply_pca(X, X)

    # Predict function
    def predict(model, X):
        return model.predict(X)

    # Prediction and evaluation
    if st.sidebar.button('Evaluate Model'):
        st.subheader(f'Evaluation Results - {model_choice}')

        # Display classification report in a table
        st.subheader('Classification Report:')
        classification_rep = load_classification_report(model_choice)
        st.text(classification_rep)

        # Display confusion matrix
        st.subheader('Confusion Matrix:')
        image_path = f'Plots/confusion_matrix_{model_choice}.png'
        try:
            image = Image.open(image_path)
            st.image(image, caption=f'Confusion Matrix - {model_choice}', use_column_width=True)
        except FileNotFoundError:
            st.error(f"Confusion matrix image for {model_choice} not found.")

    # Prediction from user input
    st.sidebar.subheader('Predict Fraudulent Transaction')
    st.sidebar.markdown("""
    Please provide the following details to predict if a transaction is fraudulent:
    - **Transaction Amount**: The amount of the transaction.
    - **Transaction Time**: The time at which the transaction occurred.
    - **Principal Component 1 (V1)**: A principal component obtained from PCA on the original features.
    - **Principal Component 2 (V2)**: Another principal component obtained from PCA on the original features.
    """)

    amount = st.sidebar.number_input('Transaction Amount', value=0.0, help="The amount of the transaction.")
    time = st.sidebar.number_input('Transaction Time', value=0.0, help="The time at which the transaction occurred.")
    v1 = st.sidebar.number_input('Principal Component 1 (V1)', value=0.0, help="A principal component obtained from PCA on the original features.")
    v2 = st.sidebar.number_input('Principal Component 2 (V2)', value=0.0, help="Another principal component obtained from PCA on the original features.")

    # Convert input features into a DataFrame
    input_features = {'Amount': [amount], 'Time': [time], 'V1': [v1], 'V2': [v2]}
    input_df = pd.DataFrame(input_features)

    if st.sidebar.button('Predict Fraud'):
        model = models[model_choice]

        # Preprocess input features similarly to training data
        input_df_processed = preprocess_data(input_df)

        # Ensure input_df_processed has the same columns as the original training data
        missing_cols = set(X.columns) - set(input_df_processed.columns)
        for col in missing_cols:
            input_df_processed[col] = 0

        # Reorder columns to match training data
        input_df_processed = input_df_processed[X.columns]

        # Apply PCA if it was used in the training
        _, input_pca = apply_pca(X, input_df_processed)

        # Make prediction
        prediction = predict(model, input_pca)

        # Display prediction result
        if prediction[0] == 1:
            st.subheader('Prediction Result: Fraudulent Transaction')
        else:
            st.subheader('Prediction Result: Legitimate Transaction')

if __name__ == '__main__':
    main()
