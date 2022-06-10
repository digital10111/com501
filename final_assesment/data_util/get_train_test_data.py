import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def get_train_test_data(train_data_file=None):
    if not train_data_file:
        df = pd.read_csv("../train_data_sq_cond.csv")
    else:
        df = pd.read_csv(train_data_file)
    df.replace("?", np.nan, inplace=True)

    # df['formability'] = df["formability"].fillna(df["formability"].mode().iloc[0])

    # df.loc[df.target == 'U', 'condition'] = df.loc[df.target == 'U', 'condition'].fillna(df["condition"].mode().iloc[0])
    # df.loc[df.target == '1', 'condition'] = df.loc[df.target == '1', 'condition'].fillna(df["condition"].mode().iloc[0])
    # df['condition'] = df.groupby('target')['condition'].apply(lambda x: x.fillna(x.mode().iat[0]))

    # df.loc[df.target == '5', 'surface_quality'] = df.loc[df.target == '5', 'surface_quality'].fillna(df["surface_quality"].mode().iloc[0])
    # df.loc[df.target == '1', 'surface_quality'] = df.loc[df.target == '1', 'surface_quality'].fillna(df["surface_quality"].mode().iloc[0])
    # df['surface_quality'] = df.groupby('target')['surface_quality'].apply(lambda x: x.fillna(x.mode().iat[0]))

    # df = df[df.columns[df.isnull().mean() < 0.37]]
    # df = df.drop(labels=["product_type"], axis=1)
    df = df[df.target != "1"]

    df['steel'].fillna(df['steel'].mode()[0], inplace=True)

    y = df.target
    X = df.drop(labels=["target"], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35, stratify=y)
    return X_train, X_test, y_train, y_test, df, X, y


if __name__ == '__main__':
    get_train_test_data()