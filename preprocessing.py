import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

def load_data(file_path):
    """Load the dataset."""
    return pd.read_csv(file_path)

def explore_additional_features(dataset):
    """Explore additional features that might influence churn."""
    # Calculate tenure (length of time as a customer)
    dataset['Tenure'] = dataset['Account length'] / 30  # Assuming 1 month = 30 days
    
    # Calculate average usage per month
    dataset['Avg usage per month'] = (dataset['Total day minutes'] + dataset['Total eve minutes'] +
                                      dataset['Total night minutes'] + dataset['Total intl minutes']) / 4
    
    # Create a feature for customer interactions with customer service
    dataset['Customer service interactions'] = dataset['Customer service calls'] > 0

def handle_missing_values(dataset):
    """Handle missing values by imputation."""
    imputer = SimpleImputer(strategy='median')  # Impute missing values with median
    dataset[numerical_features] = imputer.fit_transform(dataset[numerical_features])

def preprocess_data(dataset):
    """Preprocess the data."""
    # Encode categorical variables using one-hot encoding
    dataset = pd.get_dummies(dataset, columns=categorical_features, drop_first=True)
    
    # Split the data into features (X) and target variable (y)
    X = dataset.drop('Churn', axis=1)
    y = dataset['Churn']
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test

def build_model():
    """Build a logistic regression model."""
    # Create a pipeline for preprocessing and model training
    model_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression())
    ])
    
    return model_pipeline

def evaluate_model(model, X_test, y_test):
    """Evaluate the model."""
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1-score:", f1)
    print("ROC AUC Score:", roc_auc)

if __name__ == "__main__":
    # Constants
    file_path = 'churn-bigml-20.csv'
    categorical_features = ['State', 'International plan', 'Voice mail plan']
    numerical_features = ['Account length', 'Number vmail messages', 'Total day minutes', 'Total day calls',
                          'Total day charge', 'Total eve minutes', 'Total eve calls', 'Total eve charge',
                          'Total night minutes', 'Total night calls', 'Total night charge', 'Total intl minutes',
                          'Total intl calls', 'Total intl charge', 'Customer service calls']
    
    # Step 1: Load the dataset
    dataset = load_data(file_path)
    
    # Step 2: Explore additional features
    explore_additional_features(dataset)
    
    # Step 3: Handle missing values
    handle_missing_values(dataset)
    
    # Step 4: Preprocess the data
    X_train, X_test, y_train, y_test = preprocess_data(dataset)
    
    # Step 5: Build and train the model
    model = build_model()
    model.fit(X_train, y_train)
    
    # Step 6: Evaluate the model
    evaluate_model(model, X_test, y_test)
