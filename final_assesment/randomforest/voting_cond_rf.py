import pandas as pd
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import f1_score, classification_report, make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

from final_assesment.baselines.cross_val_custom import cross_val_score_v2
from final_assesment.baselines.decision_tree.transformers import get_categorical_pipeline, get_numerical_pipeline, \
    get_full_processeror_rfsq, get_full_processeror
from final_assesment.data_util.get_train_test_data import get_train_test_data
from final_assesment.randomforest.rf_grid_search import classification_report_with_accuracy_score

n_jobs = 40
rf1 = RandomForestClassifier(max_depth=40, n_estimators=200, n_jobs=n_jobs, oob_score=False)
rf2 = RandomForestClassifier(max_depth=30, n_estimators=200, n_jobs=n_jobs, oob_score=False)
rf3 = RandomForestClassifier(max_depth=20, n_estimators=80, n_jobs=n_jobs, oob_score=False)


clfSoft = VotingClassifier(estimators=[('rf1', rf1), ('rf2', rf2), ('rf3', rf3)], voting='hard')

X_train, y_train, X_val, y_val = get_train_test_data()

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
    ('model', clfSoft)
])

rf_pipeline.fit(X_train_stack, y_train)
y_pred = rf_pipeline.predict(X_val)
print(classification_report(y_val, y_pred, zero_division=0))  # print classification report

X_full = pd.concat([X_train, X_val], axis=0)
y_full = pd.concat([y_train, y_val], axis=0)
cv_results = cross_val_score_v2(estimator=rf_pipeline, X=X_full, y=y_full, scoring=make_scorer(classification_report_with_accuracy_score), verbose=5, n_jobs=n_jobs, cv=3)
