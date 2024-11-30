from helper.CV_Enum import CrossValidation


def perform_cross_validation(cv_method_number):
    """
    Calls the corresponding cross-validation function based on the method number.

    Parameters:
    - cv_method_number (int): The number representing the cross-validation method.
    - *args: Positional arguments for the cross-validation function.
    - **kwargs: Keyword arguments for the cross-validation function.

    Returns:
    - The output of the cross-validation function.
    """
    for cv_method in CrossValidation:
        if cv_method.number == cv_method_number:
            return cv_method.execute()
    raise ValueError(f"Invalid cross-validation method number: {cv_method_number}")


# Example: Call K-Fold CV
csv_file_path = '/path/to/your/csv_file.csv'
result = perform_cross_validation(2, csv_file_path, n_splits=5)
print(result)
