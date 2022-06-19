from sklearn.ensemble import VotingClassifier
from sklearn.metrics import f1_score, classification_report, make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

from final_assesment.baselines.cross_val_custom import cross_val_score_v2
from final_assesment.baselines.decision_tree.transformers import get_categorical_pipeline, get_numerical_pipeline, get_full_processeror_rfsq
from final_assesment.randomforest.rf_grid_search import classification_report_with_accuracy_score
from final_assesment.rf_surface_quality.get_data_rfsq import get_rfsq_train_data

n_jobs = 40
rf1 = RandomForestClassifier(n_estimators=80, n_jobs=n_jobs, oob_score=False)
rf2 = RandomForestClassifier(max_depth=30, n_estimators=10, n_jobs=n_jobs, oob_score=False)
rf3 = RandomForestClassifier(n_jobs=n_jobs, oob_score=False)

clfSoft = VotingClassifier(estimators=[('rf1', rf1), ('rf2', rf2), ('rf3', rf3)], voting='soft')

X_train, X_test, y_train, y_test, df, X, y, df_dev = get_rfsq_train_data(train_data_file="../anneal.data")


numerical_pipeline = get_numerical_pipeline(scale=True)
categorical_features = ["steel", "shape", "bore", "formability"]
numerical_features = ["carbon", "hardness", "strength", "thick", "width", "len"]
full_datapipeline = get_full_processeror_rfsq(categorical_features=categorical_features, numerical_features=numerical_features, numerical_pipeline=numerical_pipeline)
rf_pipeline = Pipeline(steps=[
    ('preprocess', full_datapipeline),
    ('model', clfSoft)
])


cv_results = cross_val_score_v2(estimator=rf_pipeline, X=X_train, y=y_train, scoring=make_scorer(classification_report_with_accuracy_score), verbose=5, n_jobs=n_jobs, cv=3)
rf_pipeline.fit(X_train, y_train)
y_test_pred = rf_pipeline.predict(X_test)
classification_report_with_accuracy_score(y_test, y_test_pred)

# df_dev[df_dev.surface_quality.isnull()]['surface_quality'] = rf_pipeline.predict(df_dev[df_dev.surface_quality.isnull()])
# df_dev.loc[df_dev.surface_quality.isnull(), 'surface_quality'] = rf_pipeline.predict(df_dev[df_dev.surface_quality.isnull()])
# # print(y_pred)
# print(df_dev['surface_quality'].isnull().any())
# df_dev.to_csv("train_data_sq_form.csv", index=False)
# k = 1