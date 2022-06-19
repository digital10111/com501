import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def get_train_test_data():
    df_train = pd.read_csv("../anneal_train.data")
    df_train.replace("?", np.nan, inplace=True)

    df_train = df_train[df_train.target != "1"]

    df_train['steel'].fillna(df_train['steel'].mode()[0], inplace=True)

    # df = df[(df.target == '2') | (df.target == '5')]
    y_train = df_train.target
    X_train = df_train.drop(labels=["target"], axis=1)

    df_val = pd.read_csv("../anneal_val.data")
    df_val.replace("?", np.nan, inplace=True)
    df_val = df_val[df_val.target != "1"]
    df_val['steel'].fillna(df_val['steel'].mode()[0], inplace=True)

    y_val = df_val.target
    X_val = df_val.drop(labels=["target"], axis=1)

    return X_train, y_train, X_val, y_val


if __name__ == '__main__':
    get_train_test_data()