from pathlib import Path
import joblib
import json
from src.preprocessing import prepare_data
from src.models import train_xgboost, train_xgboost_ros, train_balanced_rf, train_dummy
from src.evaluation import evaluate_model, find_best_threshold, evaluate_with_threshold
from src.plots import save_f1_graph

def main():
    #finds the dataset
    data_path = Path(__file__).resolve().parent/"data"/"creditcard.csv"
    #Prepares the data for training and testing
    X_train, X_val, X_test, y_train, y_val, y_test = prepare_data(data_path)
    print("\nDataset preparation completed")

    print("\nTraining:", X_train.shape)
    print("Validation:", X_val.shape)
    print("Testing:", X_test.shape)

    print("\nFraud cases:")
    print("Training:",y_train.sum())
    print("Validation:",y_val.sum())
    print("Testing:",y_test.sum())

    #This trains our first model
    print("\nTraining XGBoost:")
    model = train_xgboost(X_train, y_train)
    #See how well it works with the validation data
    evaluate_model(model, X_val, y_val)

    #train XGBoost again with balanced data
    print("\nTraining XGBoost with random oversampling:")
    model_ros = train_xgboost_ros(X_train, y_train)
    #this checks the result using the validation data
    evaluate_model(model_ros, X_val, y_val)

    #train the balanced random forest model
    print("\nTraining Balanced Random Forest:")
    model_rf = train_balanced_rf(X_train, y_train)
    #This checks how well it detects fraud
    evaluate_model(model_rf, X_val, y_val)

    #Finds a better threshold for balanced random forest
    print("\nFinding the best threshold:")
    rf_threshold = find_best_threshold(model_rf, X_val, y_val)

    #Finds the best threshold for XGBoost with ROS
    print("\nFinding the best threshold for XGBoost with random oversampling:")
    ros_threshold = find_best_threshold(model_ros, X_val, y_val)

    #Trains a simple model for comparison
    print("\nTraining Dummy Classifier:")
    dummy_model = train_dummy(X_train, y_train)
    #It checks how well it works on the validation data
    evaluate_model(dummy_model, X_val, y_val)

    #Creates the folder for the trained models
    models_folder = Path(__file__).resolve().parent/"saved_models"
    models_folder.mkdir(parents=True, exist_ok=True)
    #saves each trained model
    joblib.dump(model, models_folder/"xgboost.joblib")
    joblib.dump(model_ros, models_folder/"xgboost_ros.joblib")
    joblib.dump(model_rf, models_folder/"balanced_rf.joblib")
    joblib.dump(dummy_model, models_folder/"dummy.joblib")
    #this saves the threshold used in the final evaluation
    thresholds = {
        "xgboost": 0.5,
        "xgboost_ros": 0.5,
        "balanced_rf_default": 0.5,
        "balanced_rf_tuned": float(rf_threshold)
    }
    with open(models_folder/"thresholds.json", "w") as file:
        json.dump(thresholds, file, indent=4)
    print ("\nTrained models and threshold saved successfully")

    #Compares all our models
    models = [
        ("Dummy", dummy_model, 0.5),
        ("XGBoost",model, 0.5),
        ("XGBoost + ROS",model_ros, 0.5),
        ("Balanced RF",model_rf, 0.5),
        ("Balanced RF tuned",model_rf, rf_threshold)
    ]
    #Saves the result as a graph
    results_folder = Path(__file__).resolve().parent/"results"
    save_f1_graph(models,X_val,y_val,results_folder)

    #Tests our models using the final dataset
    print("\n==== FINAL TEST RESULTS ====")
    print("\nDummy Classifier:")
    evaluate_model(dummy_model, X_test, y_test)
    print("\nXGBoost:")
    evaluate_model(model, X_test, y_test)
    print("\nXGBoost + ROS:")
    evaluate_model(model_ros, X_test, y_test)
    print("\nBalanced RF:")
    evaluate_model(model_rf, X_test, y_test)
    print("\nBalanced RF + threshold:")
    evaluate_with_threshold(model_rf, X_test, y_test, rf_threshold)

if __name__ == "__main__":
    main()

