from matplotlib.pyplot import imshow
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import TruncatedSVD

img = mpimg.imread('dorian_grey.png')
print(type(img), img.shape)
imshow(img)
plt.show()


def rgb2gray(rgb):
    ''' Берётся среднее трёх цветов RGB'''
    tile = np.tile(np.c_[0.333, 0.333, 0.333], reps=(rgb.shape[0], rgb.shape[1], 1))
    return np.sum(tile * rgb, axis=2)


img_gray = rgb2gray(img)
print(type(img_gray), img_gray.shape)
imshow(img_gray, cmap="gray")
plt.show()

# получаем SVD разложение
svd_model = TruncatedSVD(n_components=20).fit(img_gray)
# применяем преобразование к исходным данным
X_svd = svd_model.transform(img_gray)
# трансформируем данные обратно к исходному пространству
X_svd_restored = svd_model.inverse_transform(X_svd)
# визуализируем то, что получилось
imshow(X_svd_restored, cmap="gray")
plt.show()