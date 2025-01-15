import pandas as pd
from helper.prompt import generate_qwen_prompt_with_values
from helper.qwen import get_cross_validation_technique
from helper.CV_Executor import perform_cross_validation
from helper.datatype_extraction import get_feature_data_types
import json
from dotenv import load_dotenv
import os

load_dotenv()

train_size, test_size = 60, 20

#Model name and description in your own words
model='decision trees'

#Target variable type: {"timeseries","categorical", "numerical"}
target_variable_type = 'categorical'

#Features type: {"numerical", "categorical","numerical+categorical"}
feature_type = get_feature_data_types(os.getenv('DATASET_DIRECTORY') + os.getenv('DATA_FILE_NAME'))

qwen_response = get_cross_validation_technique(
    generate_qwen_prompt_with_values(
        model,
        target_variable_type,
        feature_type))

cross_validation = json.loads(qwen_response)
result_data = perform_cross_validation(cross_validation)
print(result_data)

