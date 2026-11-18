from sklearn.metrics import(precision_score, recall_score, confusion_matrix, average_precision_score, f1_score)

def evaluate_model(model, X, y):
    #Get the model's prediction
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)[:,1]

    #This checks how well the model detects fraud
    print("\nModel results:")
    print("Precision:", round(precision_score(y, predictions, zero_division=0), 4))
    print("Recall:", round(recall_score(y, predictions, zero_division=0), 4))
    print("F1-score:", round(f1_score(y, predictions, zero_division=0), 4))
    print("Macro F1:", round(f1_score(y, predictions, average="macro", zero_division=0), 4))
    print("Macro precision:", round(precision_score(y, predictions, average="macro", zero_division=0), 4))
    print("Macro recall:", round(recall_score(y, predictions, average="macro", zero_division=0), 4))
    print("Average Precision:", round(average_precision_score(y, probabilities), 4))

    print("\nConfusion matrix:")
    print(confusion_matrix(y, predictions))

def find_best_threshold(model, X_val, y_val):
    #Gets the fraud probabilities
    probabilities = model.predict_proba(X_val)[:,1]
    best_threshold = 0.5
    best_f1 = 0
    #this tries different thresholds to find the best F1-score
    for i in range(1,100):
        threshold = i/100
        predictions = (probabilities >= threshold).astype(int)
        score = f1_score(y_val, predictions, zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_threshold = threshold
    print("\nBest threshold:", best_threshold)
    print("Best F1 score:", best_f1)
    #This shows the result using the chose threshold
    predictions = (probabilities >= best_threshold).astype(int)
    print("Precision:", round(precision_score(y_val, predictions, zero_division=0), 4))
    print("Recall:", round(recall_score(y_val, predictions, zero_division=0), 4))
    print("\nConfusion matrix:")
    print(confusion_matrix(y_val, predictions))
    return best_threshold

def evaluate_with_threshold(model, X, y, threshold):
    #Gets the fraud probabilities
    probabilities = model.predict_proba(X)[:,1]
    #use the threshold we found earlier
    predictions = (probabilities >= threshold).astype(int)
    #Checks how well the model performs
    print("\nThreshold:", threshold)
    print("Precision:", round(precision_score(y, predictions, zero_division=0), 4))
    print("Recall:", round(recall_score(y, predictions, zero_division=0), 4))
    print("F1-score:", round(f1_score(y, predictions, zero_division=0), 4))
    print("Macro F1:", round(f1_score(y, predictions, average="macro", zero_division=0), 4))
    print("Macro precision:", round(precision_score(y, predictions, average="macro", zero_division=0), 4))
    print("Macro recall:", round(recall_score(y, predictions, average="macro", zero_division=0), 4))
    print("Average Precision:", round(average_precision_score(y, probabilities), 4))
    print("\nConfusion matrix:")
    print(confusion_matrix(y, predictions))