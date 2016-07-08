from __future__ import print_function
from skimage.io import imread, imsave
from skimage import img_as_float
from sklearn.cluster import KMeans
from numpy import mean, median, reshape
from math import log10

class Clustering:
    def __init__(self, quality):
        self.quality = quality

    def fit(self, X):
        for n in range(1, 21):
            km = KMeans(n_clusters=n, init='k-means++', random_state=241)
            y = km.fit_predict(X)
            X_mean = self.compress(X, y, mean)
            X_median = self.compress(X, y, median)
            psnr_mean = self.PSNR(X, X_mean)
            psnr_median = self.PSNR(X, X_median)
            if psnr_mean > self.quality:
                image_mean = reshape(X_mean, image.shape)
                imsave('parrots-mean-'+str(n)+'.jpg', image_mean)
                return n
            if psnr_median > self.quality:
                image_median = reshape(X_median, image.shape)
                imsave('parrots-median-'+str(n)+'.jpg', image_median)
                return n

    def compress(self, X, y, func):
        C = {}
        for i in range(0, len(X)):
            if C.has_key(y[i]):
                C[y[i]].append(X[i])
            else:
                C[y[i]] = []

        value = {}
        for i in range(0, len(C)):
            r = func(map(lambda x: x[0], C[i]))
            g = func(map(lambda x: x[1], C[i]))
            b = func(map(lambda x: x[2], C[i]))
            value[i] = [r, g, b]

        return map(lambda x: value[x], y)

    def MSE(self, I, K):
        summ = 0
        for i in range(0, len(I)):
            r = (I[i][0] - K[i][0])**2
            g = (I[i][1] - K[i][1])**2
            b = (I[i][2] - K[i][2])**2
            summ += (r + g + b) / 3
        return summ / len(I)

    def PSNR(self, I, K):
        return -10 * log10(self.MSE(I, K))

image = imread('../../data/parrots.jpg')
image = img_as_float(image)

X = image.reshape((image.shape[0] * image.shape[1], image.shape[2]))

cls = Clustering(quality=20)
n = cls.fit(X)

f = open('clustering.txt', 'w')
print(n, file=f, end='')
f.close()
