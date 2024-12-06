import pandas as pd
from helper.prompt import generate_prompt_for_cv_with_mapping
from helper.qwen import get_cross_validation_technique
from helper.CV_Executor import perform_cross_validation

train_size, test_size = 60, 20

#Model name and description in your own words
model='decision trees'

#Target variable type: {"timeseries","categorical", "numerical"}
target_variable_type = 'categorical'

#Features type: {"numerical", "categorical","numerical+categorical"}
feature_type = 'categorical'

qwen_response = get_cross_validation_technique(
    generate_prompt_for_cv_with_mapping(
        model,
        target_variable_type,
        feature_type))


cross_validation_number = int(qwen_response)
result_data = perform_cross_validation(cross_validation_number)
print(result_data)

