import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def get_train_test_data(train_data_file=None):
    if not train_data_file:
        df = pd.read_csv("../../anneal.data")
    else:
        df = pd.read_csv(train_data_file)
    df.replace("?", np.nan, inplace=True)

    df = df[df.columns[df.isnull().mean() < 0.25]]
    df = df.drop(labels=["product_type"], axis=1)
    df = df[df.target != "1"]

    df['steel'].fillna(df['steel'].mode()[0], inplace=True)

    y = df.target
    X = df.drop(labels=["target"], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    return X_train, X_test, y_train, y_test, df, X, y
