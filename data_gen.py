import numpy as np
import pandas as pd

def generate_synthetic_data(n_samples=1000, n_features=5, noise=0.1, random_state=42):
    np.random.seed(random_state)
    
    # Generate random feature matrix
    X = np.random.randn(n_samples, n_features)
    
    # Generate true coefficients
    true_coefficients = np.random.randn(n_features)
    
    # Generate target variable with some noise
    y = X.dot(true_coefficients) + noise * np.random.randn(n_samples)
    
    # Create a DataFrame for the features
    feature_columns = [f'feature_{i+1}' for i in range(n_features)]
    df_features = pd.DataFrame(X, columns=feature_columns)
    
    # Add the target variable to the DataFrame
    df_features['target'] = y
    
    return df_features


def save_dataframe_to_csv(df, file_path):
    df.to_csv(file_path, index=False)
# Example usage
if __name__ == "__main__":
    df = generate_synthetic_data()
    save_dataframe_to_csv(df, 'Datasets/Regression Datasets/synthetic_data.csv')
    
    print(df.head())