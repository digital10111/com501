from sklearn.ensemble import VotingClassifier
from sklearn.metrics import f1_score, classification_report, make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

from final_assesment.baselines.cross_val_custom import cross_val_score_v2
from final_assesment.baselines.decision_tree.transformers import get_categorical_pipeline, get_numerical_pipeline, get_full_processeror_rfsq
from final_assesment.randomforest.rf_grid_search import classification_report_with_accuracy_score
from final_assesment.rf_condition.get_data_condition import get_condition_train_data

n_jobs = 40
rf1 = RandomForestClassifier(max_depth=30, n_estimators=20, n_jobs=n_jobs, oob_score=False)
rf2 = RandomForestClassifier(max_depth=10, n_estimators=60, n_jobs=n_jobs, oob_score=False)
rf3 = RandomForestClassifier(max_depth=5, n_estimators=150, n_jobs=n_jobs, oob_score=False)


clfSoft = VotingClassifier(estimators=[('rf1', rf1), ('rf2', rf2), ('rf3', rf3)], voting='soft')

X_train, X_test, y_train, y_test, df, X, y, df_dev = get_condition_train_data()


categorical_pipeline = get_categorical_pipeline()
numerical_pipeline = get_numerical_pipeline(scale=True)

categorical_features = ["steel", "shape", "bore", "surface_quality", "formability"]
numerical_features = list((set(X.columns.to_list()) - set(categorical_features)) - {'condition'})
full_datapipeline = get_full_processeror_rfsq(categorical_features, categorical_pipeline, numerical_features, numerical_pipeline)
rf_pipeline = Pipeline(steps=[
    ('preprocess', full_datapipeline),
    ('model', clfSoft)
])


cv_results = cross_val_score_v2(estimator=rf_pipeline, X=X, y=y, scoring=make_scorer(classification_report_with_accuracy_score), verbose=5, n_jobs=n_jobs, cv=3)
rf_pipeline.fit(X, y)

# df_dev[df_dev.condition.isnull()]['condition'] = rf_pipeline.predict(df_dev[df_dev.condition.isnull()])
df_dev.loc[df_dev.condition.isnull(), 'condition'] = rf_pipeline.predict(df_dev[df_dev.condition.isnull()])
# print(y_pred)
print(df_dev['condition'].isnull().any())
df_dev.to_csv("train_data_sq_cond.csv", index=False)
k = 1