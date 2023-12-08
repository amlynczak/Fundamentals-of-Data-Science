import numpy as np
import matplotlib
import pandas as pd
import matplotlib.pyplot as plt
from openml.datasets import edit_dataset, fork_dataset, get_dataset
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import plot_tree

dataset = get_dataset(1499)

X, y, categorical_indicator, attribute_names = dataset.get_data(target=dataset.default_target_attribute,include_row_id=True,)

X.describe()
print(attribute_names)

X.head()

fold = KFold(n_splits=5)
d={}

for train_index, test_index in fold.split(X):
    for metric in ["euclidean", "manhattan", "cosine"]:
        for neighbors in range(1, 10):
            accuracy = 0
            knn = KNeighborsClassifier(n_neighbors=neighbors, metric=metric)
            knn.fit(X.iloc[train_index], y.iloc[train_index])
            accuracy += knn.score(X.iloc[test_index], y.iloc[test_index])
            accuracy /= 5

            if metric in d.keys():
                d[metric].append(accuracy)
            else:
                d[metric] = [accuracy]
    
knn_results = pd.DataFrame(data=d)
knn_results

knn_results.mean()

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=10)

tree = DecisionTreeClassifier(criterion='gini',random_state=10)
tree.fit(x_train,y_train)
tree.score(x_test,y_test)

fig, ax = plt.subplots(figsize=(20,10))
plot_tree(tree, ax=ax)
plt.show()

