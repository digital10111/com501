import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import f1_score, classification_report, make_scorer
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

from final_assesment.baselines.cross_val_custom import cross_val_score_v2
from final_assesment.baselines.decision_tree.get_train_test_data import get_train_test_data
from final_assesment.baselines.decision_tree.transformers import get_categorical_pipeline, get_numerical_pipeline, \
    get_full_processeror


def fit_plot_DT(X_train_, y_train_, X_test_, y_test_, max_depth, min_leaf, min_split, min_impurity, data_pipeline):
    clf = DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=min_leaf,
                                 min_samples_split=min_split, min_impurity_decrease=min_impurity)
    dt_pipeline = Pipeline(steps=[
        ('preprocess', data_pipeline),
        ('model', clf)
    ])
    dt_pipeline.fit(X_train_, y_train_)
    y_train_pred = dt_pipeline.predict(X_train_)
    y_test_pred = dt_pipeline.predict(X_test_)
    f1 = f1_score(y_test_, y_test_pred, average='macro')  # , normalize=False)
    return f1, clf, y_test_pred, y_train_pred


def classification_report_with_accuracy_score(y_true, y_pred):
    print(classification_report(y_true, y_pred, zero_division=0))  # print classification report
    print("\n***---***\n")
    return f1_score(y_true, y_pred, average='macro')


def grid_search():
    X_train, X_test, y_train, y_test, data_df, X, y = get_train_test_data()
    dct = DecisionTreeClassifier()
    categorical_pipeline = get_categorical_pipeline()
    numerical_pipeline = get_numerical_pipeline(scale=True)

    categorical_features = ["steel", "shape", "bore"]
    numerical_features = list((set(data_df.columns.to_list()) - set(categorical_features)) - {'target'})
    full_datapipeline = get_full_processeror(categorical_features, categorical_pipeline, numerical_features,
                                             numerical_pipeline)
    dct_pipeline = Pipeline(steps=[
        ('preprocess', full_datapipeline),
        ('model', dct)
    ])

    param_grid = {
        "model__max_depth": [14, 12, 10, 8],
        "model__min_samples_leaf": [2, 3, 5, 10],
        "model__min_samples_split": [2, 4, 6, 12],
        "model__min_impurity_decrease": [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1]
    }

    print(dct_pipeline.get_params().keys())
    clf = GridSearchCV(estimator=dct_pipeline, param_grid=param_grid,
                       scoring="f1_macro", cv=5, n_jobs=20)

    cv_results = cross_val_score_v2(estimator=clf, X=X, y=y,
                                    scoring=make_scorer(classification_report_with_accuracy_score), verbose=5, n_jobs=10)

    print(numerical_features)
    print(dir(clf))


    # best_classifier = clf.best_estimator_
    #
    # print(classification_report(y_test, clf.best_estimator_.predict(X_test), zero_division=0))
    #
    # print("\n\n\n-------------TRAIN DATA METRICS---------\n\n\n")
    #
    # print(classification_report(y_train, clf.best_estimator_.predict(X_train), zero_division=0))
    #
    # plt.rcParams['figure.figsize'] = (40, 15)
    # tree.plot_tree(clf.best_estimator_)
    # plt.show()


grid_search()
