import pandas as pd
from RandomForest import (RandomForest)
from SplitData import (train_test_split)
from Metrics import (f1_score)
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
    test_size=0.2
)


# TRAIN MODEL
forest = RandomForest(
    n_trees=10,
    max_depth=5
)

forest.fit(
    X_train,
    y_train
)


# PREDICT
y_pred = forest.predict(
    X_test
)


# EVALUATE
score = f1_score(
    y_test,
    y_pred
)

print(
    "F1-score:",
    score
)