import matplotlib.pyplot as plt
from sklearn.metrics import f1_score

def save_f1_graph(models, X_val, y_val, folder):
    names = []
    scores = []
    #Gets the F1 score for each model
    for name, model, threshold in models:
        if threshold == 0.5:
            predictions = model.predict(X_val)
        else:
            probabilities = model.predict_proba(X_val)[:,1]
            predictions = (probabilities >= threshold).astype(int)
        score = f1_score(y_val, predictions, zero_division=0)
        print(name, "Threshold:", threshold, "F1:", round(score, 4))
        names.append(name)
        scores.append(score)
        #creates the graph
        plt.figure(figsize=(9, 5))
        bars = plt.bar(names,scores)
        plt.title("Fraud Detection: Validation F1-score")
        plt.ylabel("F1 Score")
        plt.ylim(0,1)
        plt.xticks(rotation=20, ha="right")
        #Add the values on top of each bar
        for bar, score in zip(bars, scores):
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                score + 0.02,
                f"{score:.3f}",
                ha="center",
            )
        #This saves the graph
        folder.mkdir(exist_ok=True)
        plt.tight_layout()
        plt.savefig(folder/"f1_comparison.png")
        plt.close()
    print("\nF1 Score graph has been saved.")
