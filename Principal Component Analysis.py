import numpy as np
from sklearn.decomposition import PCA

X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])

pca = PCA(n_components=1)
pca.fit(X)
print(pca.components_)
print(pca.explained_variance_ratio_)
