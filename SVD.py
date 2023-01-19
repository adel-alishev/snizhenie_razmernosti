import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.decomposition import TruncatedSVD

with open('eigen.pkl', 'rb') as f:
    X = pickle.load(f)

plt.plot(X[:,0], X[:,1], 'x')
plt.axis('equal')
plt.show()

# получаем SVD разложение
svd_model = TruncatedSVD(n_components=1).fit(X)
# применяем преобразование к исходным данным
X_svd = svd_model.transform(X)
# трансформируем данные обратно к исходному пространству
X_svd_restored = svd_model.inverse_transform(X_svd)
# визуализируем то, что получилось
plt.plot(X_svd_restored[:,0], X_svd_restored[:,1], 'x')
plt.show()
