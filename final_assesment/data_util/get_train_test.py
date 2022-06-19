import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def get_train_test_data(train_data_file=None):
    if not train_data_file:
        df = pd.read_csv("../anneal.data")
    else:
        df = pd.read_csv(train_data_file)
    df.replace("?", np.nan, inplace=True)

    df = df[df.target != "1"]

    y = df.target
    X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.35, stratify=y)
    X_train.to_csv("../anneal_train.data", index=False)
    X_test.to_csv("../anneal_val.data", index=False)


if __name__ == "__main__":
    get_train_test_data()