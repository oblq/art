# Test Data for FuzzyART

This directory is intended to contain test data for benchmarking and testing the FuzzyART implementation.

## MNIST Dataset

The tests are designed to use the MNIST dataset, which should be placed in this directory in CSV format.

### Required Files:
- `mnist_train.csv` - Training data from MNIST
- `mnist_test.csv` - Test data from MNIST

### Data Format:
Each CSV file should have rows in the following format:
- First column: digit label (0-9)
- Remaining columns: pixel values (0-255, will be normalized to 0-1 during loading)

### Obtaining the Data:
You can download the MNIST dataset in CSV format from various sources, such as:
- [Kaggle MNIST Dataset](https://www.kaggle.com/datasets/oddrationale/mnist-in-csv)
- Or convert the original MNIST dataset to CSV format using Python scripts

Example Python script to convert MNIST to CSV:
```python
import numpy as np
import pandas as pd
from mnist import MNIST

# Load MNIST data
mndata = MNIST('./mnist_data')
X_train, y_train = mndata.load_training()
X_test, y_test = mndata.load_testing()

# Convert to dataframes
train_df = pd.DataFrame(X_train)
train_df.insert(0, "label", y_train)

test_df = pd.DataFrame(X_test)
test_df.insert(0, "label", y_test)

# Save to CSV
train_df.to_csv("mnist_train.csv", index=False)
test_df.to_csv("mnist_test.csv", index=False)
```

### Note:
The tests will be skipped if the data files are not present.