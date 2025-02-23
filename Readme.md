# Case Studies CV

This repository contains case studies for computer vision projects.

## Data Path

The data path for the project is constructed using environment variables:

```python
import os

data_path = os.path.join(
    os.getenv('DATASET_DIRECTORY'),
    os.getenv('DATA_FILE_NAME')
)
```

Make sure to set the following environment variables before running the project:

- `DATASET_DIRECTORY`: The directory where the dataset is stored.
- `DATA_FILE_NAME`: The name of the data file.