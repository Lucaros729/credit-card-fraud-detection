# Credit Card Fraud Detection Using Machine Learning

## Project Overview

This project investigates how data balancing and decision threshold 
optimisation affect machine learning performance in credit card fraud 
detection.

The implementation compares XGBoost, XGBoost with Random Oversampling(ROS)
and Balanced Random Forest (RF) against a Dummy Classifier baseline.

This project was developed for the COMP1818 Artificial Intelligence Applications
coursework at the University Of Greenwich

## Dataset

This project uses the Credit Card Fraud Detection dataset
from the Machine Learning Group at ULB

The original dataset contains 284,807 transactions
including 492 fraudulent transactions

After removing 1,081 duplicate records, 283,726
transactions remain.


### Option 1: Dataset included in this repository
A compressed copy of the dataset is available at:
'data/creditcard.zip'

Extract it from the project's main directory using:

'''bash unzip data/creditcard.zip -d data/ '''

### Option 2: Download from the original source

The dataset is also available on Kaggle:

https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

Download and extract 'creditcard.csv' into the 'data/' folder.

Both methods provide the dataset needed to run 'main.py'.


## Project Structure

- 'main.py' - Runs the complete experiment.
- 'src/preprocessing.py' - Prepares and splits the data.
- 'src/models.py' - Trains the machine learning models.
- 'src/evaluation.py' - Calculates evaluation metrics and selects decision thresholds.
- 'src/plots.py' - Generates the comparison graph.
- 'data/' - Contains dataset download instructions.
- 'results/' - Contains the F1-score comparison graph.
- 'saved_models/' - Contains the trained models and thresholds.
- 'requirements.txt' - Lists the python dependencies for the code to be able to run.

## Installation

1. Clone or download this repository
2. Open a terminal in the project directory
3. Create a Python virtual environment:
    python3 -m venv .venv
4. Activate it on macOS or Linux:
    source .venv/bin/activate
5. Install the dependencies
    python3 -m pip install -r requirements.txt

## Running the project

From the project directory, run:
    
    python3 main.py

This trains the models, evaluates their performance, finds the decision thresholds,
saves the trained models and generates the F1-score comparison graph

## Experimental methodology

- Removed duplicate transactions.
- Excluded the Time column from the input features.
- Used stratified training, validation and test splits.
- Applied Random Oversampling only to the training data.
- Used the validation data for threshold optimisation.
- Evaluated the final models on the test data.
- Used random state 42 for reproducibility

The resulting splits contain:

- training: 181,584 transactions
- Validation: 45,396 transactions
- Testing: 56,746 transactions

## Final Test Results

| Model | Precision | Recall | F1-score |
|---|---:|---:|---:|
| Dummy Classifier | 0.0000 | 0.0000 | 0.0000 |
| XGBoost | 0.8961 | 0.7263 | 0.8023 |
| XGBoost + ROS | 0.9730 | 0.7579 | 0.8521 |
| Balanced RF | 0.1086 | 0.8526 | 0.1926 |
| Balanced RF (threshold 0.93) | 0.8933 | 0.7053 | 0.7882 |

XGBoost with ROS achieved the highest fraud-class F1-score of 85.21% on the test dataset.

Balanced RF achieved the highest recall at the default threshold, but generated more false positives.

threshold optimisation reduced these false positives while increasing the number of missed fraud cases.

## Saved Models

The 'saved_models/' folder contains:

- 'xgboost.joblib'
- 'xgboost_ros.joblib'
- 'balanced_rf.joblib'
- 'dummy.joblib'
- 'thresholds.json'

The saved files contain the trained models from the experiment.
The JSON file records the decision thresholds.

Running 'main.py' retrains and saves the models.

## References

Wijaya et al. (2024).
Comparative Analysis of Machine Learning Algorithm and Data Balancing
Techniques for Credit Card Fraud Detection.
Procedia Computer Science, 245, 677-688.

https://doi.org/10.1016/j.procs.2024.10.294