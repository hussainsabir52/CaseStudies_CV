
import pandas as pd

def get_feature_data_types(file_path):
    """
    Function to read a data file in CSV, XLSX, or XML format and return the data types of all features.
    
    Parameters:
        file_path (str): Path to the input file (CSV, XLSX, or XML).

    Returns:
        dict: A dictionary where keys are feature names and values are their data types.
    """
    # Determine file format based on extension
    file_extension = file_path.split('.')[-1].lower()

    # Read the file into a pandas DataFrame
    if file_extension == 'csv':
        df = pd.read_csv(file_path)
    elif file_extension == 'xlsx':
        df = pd.read_excel(file_path)
    elif file_extension == 'xml':
        df = pd.read_xml(file_path)
    else:
        raise ValueError("Unsupported file format. Please provide a file in CSV, XLSX, or XML format.")

    # Get data types of each column
    data_types = df.dtypes

    # Convert to dictionary with feature names as keys and data types as values
    feature_types = {col: str(dtype) for col, dtype in data_types.items()}

    return feature_types

# Example usage
file_path = "/home/mustaali-hussain/University (Master)/Semester 3/Case Studies/CaseStudies_CV/Datasets/Classification Dataset/loan_data.csv"  # Replace with your file path
feature_types = get_feature_data_types(file_path)
print(feature_types)
