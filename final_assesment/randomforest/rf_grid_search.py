from sklearn.metrics import f1_score, classification_report, make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

from final_assesment.baselines.cross_val_custom import cross_val_score_v2
from final_assesment.data_util.get_train_test_data import get_train_test_data
from final_assesment.baselines.decision_tree.transformers import get_categorical_pipeline, get_numerical_pipeline, \
    get_full_processeror, get_imputer_pipeline, get_drop_col_pipeline


def classification_report_with_accuracy_score(y_true, y_pred, **kwargs):
    # estimator = kwargs['estimator']
    print(classification_report(y_true, y_pred, zero_division=0))  # print classification report
    print("\n***---***\n")
    return f1_score(y_true, y_pred, average='macro')


def grid_search():
    n_jobs = 40
    X_train, X_test, y_train, y_test, data_df, X, y = get_train_test_data()
    rf = RandomForestClassifier(n_jobs=n_jobs, oob_score=False)

    categorical_pipeline = get_categorical_pipeline()
    numerical_pipeline = get_numerical_pipeline(scale=True)

    categorical_features = ["steel", "shape", "bore", "formability", "condition", "surface_quality"]
    numerical_features = list((set(X.columns.to_list()) - set(categorical_features)) - {'target'})
    full_datapipeline = get_full_processeror(categorical_features, categorical_pipeline, numerical_features,
                                             numerical_pipeline)
    rf_pipeline = Pipeline(steps=[
        ('preprocess', full_datapipeline),
        ('model', rf)
    ])

    param_grid = {
        "model__n_estimators": [20, 40, 60, 80, 100, 120, 150, 200],
        "model__max_depth": [5, 10, 20, 30, 40, None],

    }

    clf = GridSearchCV(estimator=rf_pipeline, param_grid=param_grid,
                       scoring="f1_macro", cv=3, n_jobs=n_jobs, return_train_score=True)

    cv_results = cross_val_score_v2(estimator=clf, X=X_train, y=y_train,
                                    scoring=make_scorer(classification_report_with_accuracy_score), verbose=5, n_jobs=n_jobs, cv=3)

    for estimator in cv_results['estimator']:
        print(estimator.best_estimator_)
    print(dir(clf))


if __name__ == '__main__':
    grid_search()
