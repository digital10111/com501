import pandas as pd

mammals = "mammals"
reptiles = "reptiles"
fish = "fish"
amphibians = "amphibians"
birds = "birds"

classes = [0, 1]


def gini_impurity(dict_class_counts):
    total = sum(dict_class_counts.values())

    accumulated_sum = 0
    for a_class in classes:
        class_count = dict_class_counts.get(a_class, 0)
        accumulated_sum += (class_count/total)**2

    return 1.0 - accumulated_sum


def attribute_split_test(data, attribute, target="animal_class", impurity='gini'):
    positive_branch = data[data[attribute] == 1][target]
    negative_branch = data[data[attribute] == 0][target]

    positive_branch_class_counts = positive_branch.value_counts().to_dict()
    negative_branch_class_counts = negative_branch.value_counts().to_dict()

    positive_branch_impurity = gini_impurity(positive_branch_class_counts)
    negative_branch_impurity = gini_impurity(negative_branch_class_counts)

    positive_branch_weight = positive_branch.shape[0]/float(data.shape[0])
    negative_branch_weight = negative_branch.shape[0] / float(data.shape[0])

    split_impurity = positive_branch_weight*positive_branch_impurity + negative_branch_weight*negative_branch_impurity
    return split_impurity, [positive_branch_class_counts, positive_branch_impurity, positive_branch_weight, negative_branch_class_counts, negative_branch_impurity, negative_branch_weight]


df = pd.read_csv("animal_class_dataset.tsv", sep="\t")
split_impurity_gb, info = attribute_split_test(df, attribute='warm-blooded')
print(split_impurity_gb)
k = 1


from sklearn.tree import DecisionTreeClassifier, plot_tree
import sklearn.tree as tree
featuers = list(set(df.columns) - {"name", "animal_class"})
clf = DecisionTreeClassifier(random_state=0)
clf.fit(df[featuers], df["animal_class"])