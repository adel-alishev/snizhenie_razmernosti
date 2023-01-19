import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
df_source = pd.read_csv('client_segmentation.csv')
X = df_source[['call_diff','sms_diff','traffic_diff']].values
y = df_source.customes_class.values

print(df_source.head())
plt.scatter(X[:,0], X[:,2])

print(f'Размерность данных: {X.shape}')
print(f'Данные {X[:10]}')

from sklearn.decomposition import PCA
pca = PCA(n_components=1).fit(X)
X_pca = pca.transform(X)
print(f'Размерность данных после PCA: {X_pca.shape}')
print(f'Данные после PCA {X_pca[:10]}')

X_new = pca.inverse_transform(X_pca)
plt.scatter(X_new[:,0], X_new[:,1])
plt.show()

X_new = pca.inverse_transform(X_pca)
plt.figure(1)
plt.subplot(211)
plt.plot(X_new[:,0], X_new[:,1], 'x')
plt.subplot(212)
plt.plot(X[:,0], X[:,1], 'o')
plt.show()