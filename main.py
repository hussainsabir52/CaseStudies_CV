import pandas as pd
from dotenv import load_dotenv
import os
from prompt import generate_prompt_with_mapping

load_dotenv()


data_path =  os.path.join(
    os.getenv('DATASET_DIRECTORY'),
    os.getenv('DATA_FILE_NAME')
)

train_size, test_size = 60, 20

#Model name and description in your own words
model='decision trees'

#Target variable type: {"timeseries","categorical", "numerical"}
target_variable_type = 'categorical'

#Features type: {"numerical", "categorical","numerical+categorical"}
feature_type = 'categorical'



print(generate_prompt_with_mapping(model, target_variable_type, feature_type))