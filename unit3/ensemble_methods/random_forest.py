import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier


n_estimators = 50
cmap = plt.cm.RdYlBu

plot_step = 0.02
plot_step_coarser = 0.5
RANDOM_SEED = 13

titanic = pd.read_csv('./train.csv')
titanic = titanic[[ 'Age', 'SibSp', 'Fare', 'Pclass', 'Survived']]
iris = load_iris()

titan = np.array(titanic)

# dealing with missingness
titan[titanic.isnull()]=0


def plot_boundary(X, y, plot_idx, models, model, model_title):
    plt.subplots(5, 2, plot_idx)
    if plot_idx <= len(models):
        plt.title(model_title, fontsize=9)

    # Now plot the decision boundary using a fine mesh as input to a filled contour plot

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 0].max() + 1

    xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step), np.arange(y_min, y_max, plot_step))

    if isinstance(model, DecisionTreeClassifier):
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        cs = plt.contourf(xx, yy, Z, cmap=cmap)
    else:
        estimator_alpha = 1.0/len(model.estimators_)
        for tree in model.estimators_:
            Z = tree.predict(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)
            cs = plt.contourf(xx, yy, Z, alpha=estimator_alpha, cmap=cmap)

    pass


def apply_DecTree_RandForest(dataname = 'titan'):
    plot_idx = 1

    models = [DecisionTreeClassifier(max_depth=None), RandomForestClassifier(n_estimators=n_estimators)]
    plt.figure(figsize=(12, 10), dpi=80)

    for pair in ([0, 1], [0, 2], [2, 3], [1,2], [1,3]):
        for model in models:
            if dataname == 'iris':
                X = iris.data[:, pair]
                y = iris.target
            else:
                X = titanic.data[:, pair]
                y = titanic[:, -1]

            idx = np.arange(X.shape[0])
            np.random.seed(RANDOM_SEED)
            np.random.shuffle(idx)

            # mean = X.mean(axis=0)
            # std = X.std(axis=0)
            # X = (X-mean)/std

            model.fit(X, y)

            scores = model.score(X, y)

            model_title = str(type(model)).split(".")[-1][:-2][:-len("Classifier")]

            model_details = model_title
            model_details += " with features {} has a score of {}".format(pair, scores)

            if hasattr(model, "estimators_"):
                model_details += " with {} estimators".format(len(model.estimators_))

            plot_it = True

            if plot_it:
                plot_boundary(X, y, plot_idx, models, model, model_title)
                plot_idx += 1




