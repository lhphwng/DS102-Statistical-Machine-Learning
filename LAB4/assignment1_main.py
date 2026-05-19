import numpy as np
import pandas as pd
from DecisionTree import (DecisionTree)
from SplitData import (train_test_split)
from Metrics import (f1_score)
from DataLoader import (load_wine_data)


# LOAD DATASET
X, y = load_wine_data()

# SPLIT DATA
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
tree = DecisionTree(max_depth=5)

tree.fit(
    X_train,
    y_train
)


# PREDICT
y_pred = tree.predict(X_test)


# EVALUATE
score = f1_score(y_test, y_pred)

print("F1-score:", score)