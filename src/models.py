from xgboost import XGBClassifier
from imblearn.over_sampling import RandomOverSampler
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.dummy import DummyClassifier

def train_xgboost(X_train, y_train):
    #This will create the first XGBoost model
    model = XGBClassifier(
        n_estimators=100,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42
    )
    #This will train the model using our training data
    model.fit(X_train, y_train)
    return model

def train_xgboost_ros(X_train, y_train):
    #balance the training data
    ros = RandomOverSampler(random_state=42)
    X_balanced, y_balanced = ros.fit_resample(X_train, y_train)
    print("Training examples after balancing:", len(y_balanced))
    #train XGBoost with the balanced data
    model = train_xgboost(X_balanced, y_balanced)
    return model

def train_balanced_rf(X_train, y_train):
    #this creates the balanced random forest model
    model = BalancedRandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )
    #Trains the model with our training data
    model.fit(X_train, y_train)

    return model

def train_dummy(X_train, y_train):
    #Creates a simple model that always predicts the most common class
    model = DummyClassifier(strategy="most_frequent")
    #Fits the model using our training data
    model.fit(X_train, y_train)
    return model
