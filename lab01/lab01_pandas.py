import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt

# Load the dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/haberman/haberman.data"
column_names = ["Age", "Year_of_operation", "Nodes_detected", "Survival_status"]
df = pd.read_csv(url, header=None, names=column_names)

# Drop the last column (class label) for outlier detection
data_for_detection = df.drop("Survival_status", axis=1)

# Task 1: Draw three-dimensional group scatter plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

colors = {1: 'blue', 2: 'red'}
for status, color in colors.items():
    subset = df[df["Survival_status"] == status]
    ax.scatter(subset["Age"], subset["Year_of_operation"], subset["Nodes_detected"], c=color, label=f'Survival {status}')

ax.set_xlabel("Age")
ax.set_ylabel("Year_of_operation")
ax.set_zlabel("Nodes_detected")
plt.legend()
plt.show()

# Task 2: Univariate outlier detection using mean and standard deviation
outliers_univariate = []
for column in data_for_detection.columns:
    mean = data_for_detection[column].mean()
    std = data_for_detection[column].std()
    lower_bound = mean - 3 * std
    upper_bound = mean + 3 * std

    outliers_univariate.extend(data_for_detection[(data_for_detection[column] < lower_bound) | (data_for_detection[column] > upper_bound)].index)

outliers_univariate = list(set(outliers_univariate))
print(f"Univariate Outliers (3-sigma method): {len(outliers_univariate)}")

# Task 3: Multivariate outlier detection using k=5
k = 5
nn = NearestNeighbors(n_neighbors=k)
nn.fit(data_for_detection)
distances, indices = nn.kneighbors(data_for_detection)

outliers_multivariate = indices[:, -1]
print(f"Multivariate Outliers (k=5): {len(outliers_multivariate)}")

# Task 4: Compare results
common_outliers = set(outliers_univariate).intersection(outliers_multivariate)
print(f"Common Outliers: {len(common_outliers)}")
