import pandas as pd
from imblearn.over_sampling import SMOTENC
from sklearn.metrics import f1_score, classification_report, make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from statistics import variance, mean
from final_assesment.baselines.cross_val_custom import cross_val_score_v2
from final_assesment.data_util.get_train_test_data import get_train_test_data
from final_assesment.baselines.decision_tree.transformers import get_categorical_pipeline, get_numerical_pipeline, \
    get_full_processeror, get_imputer_pipeline, get_drop_col_pipeline
from collections import defaultdict


cross_val_F1_means = []
cross_val_F1_vars = []
val_set_F1 = []
estimator_count = defaultdict(int)


def classification_report_with_accuracy_score(y_true, y_pred, **kwargs):
    # estimator = kwargs['estimator']
    print(classification_report(y_true, y_pred, zero_division=0))  # print classification report
    print("F1_score: ", f1_score(y_true, y_pred, average='macro'))
    print("\n***---***\n")

    return f1_score(y_true, y_pred, average='macro')


def grid_search():
    n_jobs = -1
    X_train, y_train, X_val, y_val = get_train_test_data()
    rf = RandomForestClassifier(n_jobs=n_jobs, oob_score=False)

    # categorical_pipeline = get_categorical_pipeline()
    numerical_pipeline = get_numerical_pipeline(scale=True)
    all_features = X_train.columns.to_list()
    numerical_features = ["carbon", "hardness", "strength", "thick", "width", "len"]
    all_categorical = list(set(all_features) - set(numerical_features))
    categorical_features = ["steel", "shape", "bore", "formability", "condition", "surface_quality"]
    na_imputer = ['bl', 'temper_rolling', 'bf', 'non_ageing', 'cbond', 'bt', 'lustre', 'ferro', 'chrom', 'phos']
    drop_categorical_features = set(all_categorical) - set(categorical_features) - set(na_imputer)
    X_train_stack = pd.concat([X_train[all_categorical], X_train[numerical_features]], axis=1)

    full_datapipeline = get_full_processeror(all_categorical_features=all_categorical,
                                             cat_features_to_drop=drop_categorical_features,
                                             numerical_pipeline=numerical_pipeline,
                                             numerical_features=numerical_features,
                                             na_imputer_drop=na_imputer)
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

    cv_results = cross_val_score_v2(estimator=clf, X=X_train_stack, y=y_train,
                                    scoring=make_scorer(classification_report_with_accuracy_score), verbose=5, n_jobs=n_jobs, cv=3)

    run_mean = mean(cv_results['test_score'])
    run_var = variance(cv_results['test_score'])
    print("RUN: ", run_mean, run_var)
    cross_val_F1_means.append(run_mean)
    cross_val_F1_vars.append(run_var)

    for estimator in cv_results['estimator']:
        estimator_count[(estimator.best_estimator_.named_steps.model.max_depth, estimator.best_estimator_.named_steps.model.n_estimators)] += 1

    # rf_pipeline.fit(X_train, y_train)
    # y_val_predict = rf_pipeline.predict(X_val)

    # print("Val Classi")
    # F1_val = classification_report_with_accuracy_score(y_val, y_val_predict)
    # val_set_F1.append(F1_val)


if __name__ == '__main__':
    for i in range(30):
        grid_search()
        print("OVERALL: ", mean(cross_val_F1_means), mean(cross_val_F1_vars))
        print(estimator_count)
        print("*"*10)
        # print("Overall Val: ", mean(val_set_F1), variance(val_set_F1))
