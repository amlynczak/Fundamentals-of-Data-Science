from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


data = pd.read_csv("seeds.csv", header=None)
#print(data)

pca = PCA(n_components=2)
data_red = pca.fit_transform(data)

pca_var = PCA(n_components=7)
pca_var.fit_transform(data)

PC_values = np.arange(pca_var.n_components_) + 1
plt.bar(PC_values, pca_var.explained_variance_ratio_, color='blue')
plt.title('Scree Plot')
plt.xlabel('Principal Component')
plt.ylabel('Variance Explained')
plt.show()

#print(data_red)
i = 0

for el in data[7]:
    if el == 1:
        clr = 'red'
    elif el == 2:
        clr = 'blue'
    elif el == 3:
        clr = 'yellow'

    plt.scatter(data_red[i][0], data_red[i][1], marker= 'o', color = clr)
    i = i+1

plt.show()

pca_1 = PCA(n_components=3)
data_red_1 = pca_1.fit_transform(data)
fig = plt.figure(figsize=(10, 8))
plt3 = fig.add_subplot(111, projection='3d')
i = 0
for el in data[7]:
    if el == 1:
        clr = 'red'
    elif el == 2:
        clr = 'blue'
    elif el == 3:
        clr = 'yellow'

    plt3.scatter(data_red_1[i][0], data_red_1[i][1], data_red_1[i][2], marker= 'o', color = clr)
    i = i+1

plt.show()
