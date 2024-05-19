import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency, ttest_ind
from sklearn.impute import SimpleImputer

def load_data(file_path):
    """Load the dataset."""
    return pd.read_csv(file_path)

def handle_missing_values(dataset, strategy='median'):
    """Handle missing values using imputation."""
    imputer = SimpleImputer(strategy=strategy)
    dataset[numerical_features] = imputer.fit_transform(dataset[numerical_features])

def explore_categorical_variables(dataset):
    """Explore categorical variables and visualize their distributions."""
    for var in categorical_features:
        plt.figure(figsize=(8, 6))
        sns.countplot(x=var, data=dataset, palette='Set2')
        plt.title(f'{var} Distribution')
        plt.xlabel(var)
        plt.xticks(rotation=90)
        plt.ylabel('Count')
        plt.show()
        
        # Perform chi-square test for independence
        contingency_table = pd.crosstab(dataset[var], dataset['Churn'])
        chi2, p, _, _ = chi2_contingency(contingency_table)
        print(f"Chi-square test p-value for '{var}' vs. 'Churn': {p}")

def visualize_numerical_features(dataset):
    """Visualize numerical features using boxplots."""
    for feature in numerical_features:
        plt.figure(figsize=(8, 6))
        sns.boxplot(x='Churn', y=feature, data=dataset, palette='Set1')
        plt.title(f'Distribution of {feature} by Churn')
        plt.xlabel('Churn')
        plt.ylabel(feature)
        plt.show()
        
        # Perform t-test for difference of means
        churn_yes = dataset[dataset['Churn'] == 1][feature]
        churn_no = dataset[dataset['Churn'] == 0][feature]
        _, p = ttest_ind(churn_yes, churn_no, equal_var=False)
        print(f"T-test p-value for difference in means of '{feature}' between Churn = Yes and Churn = No: {p}")

def visualize_relationships(dataset):
    """Visualize relationships between variables using scatter plots and pair plots."""
    # Scatter plot
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x='Total day minutes', y='Total eve minutes', hue='Churn', data=dataset, palette='coolwarm')
    plt.title('Scatter plot: Total day minutes vs. Total eve minutes')
    plt.xlabel('Total day minutes')
    plt.ylabel('Total eve minutes')
    plt.show()
    
    # Pair plot
    sns.pairplot(dataset[numerical_features + ['Churn']], hue='Churn', palette='husl')
    plt.title('Pair plot')
    plt.show()

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
    
    # Step 2: Handle missing values
    handle_missing_values(dataset)
    
    # Step 3: Explore categorical variables
    explore_categorical_variables(dataset)
    
    # Step 4: Visualize numerical features
    visualize_numerical_features(dataset)
    
    # Step 5: Visualize relationships between variables
    visualize_relationships(dataset)
