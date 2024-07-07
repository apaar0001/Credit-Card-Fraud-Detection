import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestCentroid

# Function to load the dataset
def load_dataset(file_path):
    df = pd.read_csv(file_path)
    return df

# Function for data exploration
def explore_data(df):
    print("First few rows of the dataset:")
    print(df.head())
    
    print("\nDataset Information:")
    print(df.info())
    
    print("\nSummary Statistics:")
    print(df.describe())
    
    print("\nMissing Values:")
    print(df.isnull().sum())
    
    # Plot the distribution of the target variable 'Class'
    plt.figure(figsize=(6, 4))
    sns.countplot(x='Class', data=df)
    plt.title('Distribution of Fraudulent and Non-Fraudulent Transactions')
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.savefig('plots/class_distribution.png')
    plt.close()
    
    # Plot the distribution of the 'Amount' feature
    plt.figure(figsize=(6, 4))
    sns.histplot(df['Amount'], bins=50, kde=True)
    plt.title('Distribution of Transaction Amounts')
    plt.xlabel('Amount')
    plt.ylabel('Frequency')
    plt.savefig('plots/transaction_amount_distribution.png')
    plt.close()
    
    # Plot the correlation matrix
    plt.figure(figsize=(12, 10))
    correlation_matrix = df.corr()
    sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', linewidths=0.2)
    plt.title('Correlation Matrix')
    plt.savefig('plots/correlation_matrix.png')
    plt.close()
    
    # Plot the distribution of 'Time' feature for fraudulent and non-fraudulent transactions
    plt.figure(figsize=(6, 4))
    sns.histplot(df[df['Class'] == 0]['Time'], bins=50, kde=True, color='blue', label='Non-Fraudulent')
    sns.histplot(df[df['Class'] == 1]['Time'], bins=50, kde=True, color='red', label='Fraudulent')
    plt.title('Distribution of Transaction Time')
    plt.xlabel('Time')
    plt.ylabel('Frequency')
    plt.legend()
    plt.savefig('plots/transaction_time_distribution.png')
    plt.close()

# Function for data preprocessing
def preprocess_data(df):
    scaler = StandardScaler()
    df['Scaled_Amount'] = scaler.fit_transform(df['Amount'].values.reshape(-1, 1))
    df.drop(['Amount', 'Time'], axis=1, inplace=True)
    return df

# Function for feature engineering (PCA)
def apply_pca(X_train, X_test):
    pca = PCA(n_components=2, random_state=42)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    print("\nExplained Variance Ratio after PCA:")
    print(pca.explained_variance_ratio_)
    return X_train_pca, X_test_pca

# Function for model training and evaluation
def train_evaluate_model(classifier, classifier_name, X_train, X_test, y_train, y_test):
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    
    # Model evaluation
    print("\nClassifier:", classifier_name)
    classification_rep = classification_report(y_test, y_pred, output_dict=True)
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    
    conf_matrix = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix - ' + classifier_name)
    plt.savefig(f'plots/confusion_matrix_{classifier_name}.png')
    plt.close()
    
    roc_auc = roc_auc_score(y_test, y_pred)
    print("ROC AUC Score:", roc_auc)
    
    # Save classification report to a text file
    with open(f'results/classification_report_{classifier_name}.txt', 'w') as f:
        f.write(f'Classification Report for {classifier_name}:\n\n')
        for key in classification_rep:
            f.write(f'{key}:\n{classification_rep[key]}\n\n')

    # Save the trained model
    joblib.dump(classifier, f'{classifier_name}_model.joblib')

# Function to handle the entire workflow
def main():
    # Create directories to store results if they don't exist
    if not os.path.exists('Plots'):
        os.makedirs('Plots')
    if not os.path.exists('Results'):
        os.makedirs('Results')
    
    # Step 1: Load the dataset
    file_path = 'Datasets\creditcard.csv'  # Adjust path as necessary
    df = load_dataset(file_path)
    
    # Step 2: Data exploration
    explore_data(df)
    
    # Step 3: Data preprocessing
    df = preprocess_data(df)
    
    # Step 4: Handle class imbalance and split data
    X = df.drop('Class', axis=1)
    y = df['Class']
    
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    
    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)
    
    # Step 5: Feature engineering (PCA - Optional)
    X_train_pca, X_test_pca = apply_pca(X_train, X_test)
    
    # Step 6: Model training and evaluation
    classifiers = {
        'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
        'GradientBoosting': GradientBoostingClassifier(random_state=42),
        'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42),
        'NearestCentroid': NearestCentroid()
    }
    
    for clf_name, clf in classifiers.items():
        train_evaluate_model(clf, clf_name, X_train_pca, X_test_pca, y_train, y_test)

if __name__ == "__main__":
    main()
