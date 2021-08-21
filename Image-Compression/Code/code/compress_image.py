import os
from PIL import Image
import numpy as np


def init_K_centroids(X, K):
    m = len(X)
    return X[np.random.choice(m, K, replace=False), :]


def find_closest_centroids(X, centroids):
    m = len(X)
    c = np.zeros(m)
    for i in range(m):
        distances = np.linalg.norm(X[i] - centroids, axis=1)
        c[i] = np.argmin(distances)
    return c


def find_means(X, idx, K):
    _, n = X.shape
    centroids = np.zeros((K, n))
    for k in range(K):
        examples = X[np.where(idx == k)]
        mean = [np.mean(column) for column in examples.T]
        centroids[k] = mean
    return centroids


def find_k_means(X, K):
    centroids = init_K_centroids(X, K)
    previous_centroids = centroids
    for _ in range(10):
        idx = find_closest_centroids(X, centroids)
        centroids = find_means(X, idx, K)
        if (centroids == previous_centroids).all():
            return centroids
        else:
            previous_centroids = centroids
    return centroids, idx


def load_image(path):
    image = Image.open(path)
    return np.asarray(image) / 255


try:
    image_base_path_input = "../inputs/"
    image_base_path_output = "../outputs/compressed_"
    file_name = input("Please type your image name (with extension) for your image which is in inputs folder:\n")
    image_path = image_base_path_input + file_name
    assert os.path.isfile(image_path)
except (IndexError, AssertionError):
    print('File not found!')

image = load_image(image_path)
w, h, d = image.shape
print('Image dimensions:\n\thas Width = {}, Height = {}, Depth = {}'.format(w, h, d))

X = image.reshape((w * h, d))
K = 20
colors, _ = find_k_means(X, K)

idx = find_closest_centroids(X, colors)

idx = np.array(idx, dtype=np.uint8)
raw_image = np.array(colors[idx, :] * 255, dtype=np.uint8).reshape((w, h, d))

result = Image.fromarray(raw_image)

result.save(image_base_path_output + file_name)
print("Done\nImage is save in outputs folder")
