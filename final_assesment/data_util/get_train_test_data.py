import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def get_train_test_data():
    df_train = pd.read_csv("../anneal.data")
    df_train.replace("?", np.nan, inplace=True)

    df_train = df_train[df_train.target != "1"]

    df_train['steel'].fillna(df_train['steel'].mode()[0], inplace=True)
    df_train.enamelability = df_train.enamelability.astype(float)

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


def get_test_data_fin():
    df_train = pd.read_csv("../anneal_train.data")
    df_train.replace("?", np.nan, inplace=True)
    df_train = df_train[df_train.target != "1"]


    df_test = pd.read_csv("../anneal.test")
    df_test.replace("?", np.nan, inplace=True)
    df_test.enamelability = df_test.enamelability.astype(float)
    df_test = df_test[df_test.target != "1"]
    df_test['steel'].fillna(df_train['steel'].mode()[0], inplace=True)

    y_test = df_test.target
    X_test = df_test.drop(labels=["target"], axis=1)

    return X_test, y_test


if __name__ == '__main__':
    get_train_test_data()