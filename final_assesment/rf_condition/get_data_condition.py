import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def get_condition_train_data(train_data_file=None):
    if not train_data_file:
        df = pd.read_csv("../train_data_sq_form.csv")
    else:
        df = pd.read_csv(train_data_file)
    df.replace("?", np.nan, inplace=True)

    df = df[df.target != "1"]
    df['steel'].fillna(df['steel'].mode()[0], inplace=True)
    df['formability'] = df["formability"].fillna(df["formability"].mode().iloc[0]).astype(int)
    # df = df[df.columns[df.isnull().mean() < 0.80]]
    df_dev = df.copy(deep=True)

    df = df[df.condition.notnull()]
    y = df.condition
    X = df.drop(labels=["condition", "target"], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, stratify=y)
    return X_train, X_test, y_train, y_test, df, X, y, df_dev


if __name__ == '__main__':
    get_condition_train_data()












    # df_predict = df[df.surface_quality.isnull()]
    # df_predict = df_predict[df_predict.target != "1"]
    # df_predict.drop(labels=["surface_quality", "target"], axis=1)
    # df['formability'] = df["formability"].fillna(df["formability"].mode().iloc[0])

    # df.loc[df.target == 'U', 'condition'] = df.loc[df.target == 'U', 'condition'].fillna(df["condition"].mode().iloc[0])
    # df.loc[df.target == '1', 'condition'] = df.loc[df.target == '1', 'condition'].fillna(df["condition"].mode().iloc[0])
    # df['condition'] = df.groupby('target')['condition'].apply(lambda x: x.fillna(x.mode().iat[0]))

    # df.loc[df.target == '5', 'surface_quality'] = df.loc[df.target == '5', 'surface_quality'].fillna(df["surface_quality"].mode().iloc[0])
    # df.loc[df.target == '1', 'surface_quality'] = df.loc[df.target == '1', 'surface_quality'].fillna(df["surface_quality"].mode().iloc[0])
    # df['surface_quality'] = df.groupby('target')['surface_quality'].apply(lambda x: x.fillna(x.mode().iat[0]))