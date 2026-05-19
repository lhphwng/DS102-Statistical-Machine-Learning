import pandas as pd

def load_wine_data():
    # LOAD DATA
    red_df = pd.read_csv(
        "wine+quality/winequality-red.csv",
        sep=";"
    )

    white_df = pd.read_csv(
        "wine+quality/winequality-white.csv",
        sep=";"
    )

    # ADD TYPE FEATURE
    red_df["type"] = 0
    white_df["type"] = 1

    
    # CONCAT DATA
    df = pd.concat(
        [red_df, white_df],
        axis=0
    )

    # SHUFFLE
    df = df.sample(
        frac=1,
        random_state=42
    ).reset_index(drop=True)


    # FEATURES / LABELS
    X = df.drop(
        "quality",
        axis=1
    ).values

    y = df["quality"].values


    # BINARY CLASSIFICATION
    y = (
        y >= 6
    ).astype(int)

    return X, y