import numpy as np
from sklearn.datasets import  load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV


def get_data():
    X, y = load_iris(return_X_y=True)
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25)
    return X_train, X_val, X_test, X_train_val, y_train, y_val, y_test, y_train_val


def one_fold_cv():
    X_train, X_val, X_test, X_train_val, y_train, y_val, y_test, y_train_val = get_data()
    depths = [2, 5, 10, None]
    errors = []

    for max_depth in depths:
        rf = RandomForestClassifier(max_depth=max_depth)
        rf.fit(X_train, y_train)
        error = np.mean(rf.predict(X_val) != y_val)
        errors.append(error)

    best_depth = depths[np.argmax(errors)]

    rf = RandomForestClassifier(max_depth=best_depth)
    rf.fit(X_train_val, y_train_val)

    test_error = np.mean(rf.predict(X_test) != y_test)

    return test_error


def k_fold_cv():
    X_train, X_val, X_test, X_train_val, y_train, y_val, y_test, y_train_val = get_data()
    p_grid = {'max_depth': [2, 5, 10, None]}

    rf = RandomForestClassifier()
    clf = GridSearchCV(estimator=rf, param_grid=p_grid)
    clf.fit(X_train, y_train)

    rf = RandomForestClassifier(**clf.best_params_)
    rf.fit(X_train_val, y_train_val)
    test_error = np.mean(rf.predict(X_test) != y_test)



