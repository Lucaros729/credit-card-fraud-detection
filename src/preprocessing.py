import pandas as pd
from sklearn.model_selection import train_test_split

def prepare_data(file_path):
    #Loads the dataset
    df = pd.read_csv(file_path)
    #This removes any duplicate rows just to keep the data clean
    original_size = len(df)
    df = df.drop_duplicates()
    print("Duplicates removed:",original_size - len(df))
    #This splits the data into features(x) and target(y)
    #I'm removing Time because the paper did that too
    X = df.drop(columns=["Class", "Time"])
    y = df["Class"]
    # First split: 80% will go for development and 20% for final testing
    X_dev, X_test, y_dev, y_test = train_test_split(
X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Second split: take part of the development for validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_dev,
        y_dev,
        test_size=0.2,
        random_state=42,
        stratify=y_dev
    )

    return X_train, X_val, X_test, y_train, y_val, y_test