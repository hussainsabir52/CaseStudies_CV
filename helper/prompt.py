def generate_prompt_with_mapping(model, target_variable_type, feature_type):
    cross_validation_mapping = {
        1: "Hold-Out Cross-Validation",
        2: "K-Fold Cross-Validation",
        3: "Leave-One-Out Cross-Validation (LOOCV)",
        4: "Leave-P-Out Cross-Validation",
        5: "Stratified K-Fold Cross-Validation",
        6: "Repeated K-Fold Cross-Validation",
        7: "Nested K-Fold Cross-Validation",
        8: "Rolling Window Partition",
    }

    prompt = f""" 
            Given the following context, choose a number between 1 and 10, where each number corresponds to a unique cross-validation technique. 
            Your task is to suggest the most suitable cross-validation method based on the provided details:

            - **Model**: {model}  
            - **Target Variable Type**: {target_variable_type}  
            - **Feature Type**: {feature_type}  

            Here is the mapping of numbers to cross-validation techniques:
            {cross_validation_mapping}

            Please choose the number that best represents your choice of cross-validation technique.
            
            Result should only be a number for e.g "1"
            """
    return prompt


