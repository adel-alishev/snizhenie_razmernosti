import numpy as np
import matplotlib.pyplot as plt
import pickle

with open('eigen.pkl', 'rb') as f:
    X = pickle.load(f)

plt.plot(X[:,0], X[:,1], 'x')
plt.axis('equal')
plt.show()

print(f'Размерность данных: {X.shape}')
print(f'Данные {X[:10]}')

from sklearn.decomposition import PCA
pca = PCA(n_components=1).fit(X)
X_pca = pca.transform(X)
print(f'Размерность данных после PCA: {X_pca.shape}')
print(f'Данные после PCA {X_pca[:10]}')

X_new = pca.inverse_transform(X_pca)
plt.figure(1)
plt.subplot(211)
plt.plot(X_new[:,0], X_new[:,1], 'x')
plt.subplot(212)
plt.plot(X[:,0], X[:,1], 'o')
plt.show()