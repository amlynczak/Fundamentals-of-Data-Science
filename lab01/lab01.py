import numpy as np
import matplotlib.pyplot as plt
import statistics
from sklearn.neighbors import NearestNeighbors

age_all = []
year_of_operation_all = []
nodes_detected_all = []

age_1 = []
year_of_operation_1 = []
nodes_detected_1 = []

age_2 = []
year_of_operation_2 = []
nodes_detected_2 = []

data = open("haberman.data", "r")

Lines = data.readlines()

#TASK 1
for line in Lines:
    tmp = line.split(",")
    if int(tmp[3]) == 1:
        age_1.append(int(tmp[0]))
        year_of_operation_1.append(int(tmp[1]))
        nodes_detected_1.append(int(tmp[2]))
    if int(tmp[3]) == 2:
        age_2.append(int(tmp[0]))
        year_of_operation_2.append(int(tmp[1]))
        nodes_detected_2.append(int(tmp[2]))
    age_all.append(int(tmp[0]))
    year_of_operation_all.append(int(tmp[1]))
    nodes_detected_all.append(int(tmp[2]))


fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(age_1, year_of_operation_1, nodes_detected_1, c='blue')
ax.scatter(age_2, year_of_operation_2, nodes_detected_2, c='red')

ax.set_xlabel("Age")
ax.set_ylabel("Year_of_operation")
ax.set_zlabel("Nodes_detected")
plt.legend()
plt.show()

#TASK 2

def detect_outliers(attribute, attribute_name):
    mean = np.mean(attribute)
    std = np.std(attribute)
    lower_bound = mean - 3 * std
    upper_bound = mean + 3 * std

    outliers = [index for index, value in enumerate(attribute) if value < lower_bound or value > upper_bound]
    
    print(f"Outliers for {attribute_name}: {len(outliers)}")
    print(f"Indices of outliers: {outliers}")

    return outliers

# Detect outliers for each attribute
outliers_age = detect_outliers(age_all, "Age")
outliers_year = detect_outliers(year_of_operation_all, "Year_of_operation")
outliers_nodes = detect_outliers(nodes_detected_all, "Nodes_detected")

# TASK 3
# TASK 3: Distance-based multivariate outlier detection
def detect_multivariate_outliers(features, k):
    neighbors = NearestNeighbors(n_neighbors=k)
    neighbors.fit(features)
    distances, _ = neighbors.kneighbors(features)

    # Distance to the kth neighbor for each data point
    distance_to_kth_neighbor = distances[:, -1]

    # Find outliers
    outliers = np.argsort(distance_to_kth_neighbor)[-len(features):]

    print(f"Multivariate Outliers: {len(outliers)}")
    print(f"Indices of outliers: {outliers}")

    return outliers

# Combine features into a matrix
features_all = np.column_stack((age_all, year_of_operation_all, nodes_detected_all))

# Set k to 5 as mentioned in the task description
k_value = 5
outliers_multivariate = detect_multivariate_outliers(features_all, k_value)

# Print the indices of multivariate outliers
print("Multivariate Outliers Indices:", outliers_multivariate)

# TASK 4: Compare the results of univariate and multivariate outlier detection
common_outliers = set(outliers_age) & set(outliers_year) & set(outliers_nodes) & set(outliers_multivariate)
print(f"Common Outliers between Univariate and Multivariate Methods: {len(common_outliers)}")
