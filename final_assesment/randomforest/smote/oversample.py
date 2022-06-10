from collections import Counter
from imblearn.over_sampling import SMOTENC
from final_assesment.data_util.get_train_test_data import get_train_test_data

X_train, X_test, y_train, y_test, data_df, X, y = get_train_test_data(train_data_file="../../anneal.data")


sm = SMOTENC(categorical_features=[0, 4, 8])
X_res, y_res = sm.fit_resample(X_train, y_train)
print(f'Resampled dataset samples per class {Counter(y_res)}')