import pandas as pd
from sklearn.model_selection import (train_test_split)
from sklearn.tree import (DecisionTreeClassifier)
from sklearn.ensemble import (RandomForestClassifier)
from sklearn.metrics import (f1_score)
from DataLoader import (load_wine_data)


# LOAD DATASET
X, y = load_wine_data()

# TRAIN TEST SPLIT
(
    X_train,
    X_test,
    y_train,
    y_test
) = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# DECISION TREE
decision_tree = DecisionTreeClassifier(
    max_depth=5,
    random_state=42
)

decision_tree.fit(
    X_train,
    y_train
)

dt_predictions = decision_tree.predict(
    X_test
)

dt_score = f1_score(
    y_test,
    dt_predictions
)

print(
    "Decision Tree F1-score:",
    dt_score
)


# RANDOM FOREST
random_forest = RandomForestClassifier(
    n_estimators=100,
    max_depth=5,
    random_state=42
)

random_forest.fit(
    X_train,
    y_train
)

rf_predictions = random_forest.predict(
    X_test
)

rf_score = f1_score(
    y_test,
    rf_predictions
)

print(
    "Random Forest F1-score:",
    rf_score
)