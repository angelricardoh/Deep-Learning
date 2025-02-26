import sys
import mmap
import numpy as np
import os
from os import path
# import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from numpy import linalg as LA

IMAGE_SCOPE_SIZE = 800

def KNN(k, X, y, x):
    N, _ = X.shape
    num_classes = len(np.unique(y))
    distances = np.zeros((N,1))
    for i in range(N):
        distances[i] = np.linalg.norm(X[i] - x)

    flatten_distances = distances.flatten()

    votes = np.zeros(num_classes, dtype=np.float32)
    neighbors = sorted(flatten_distances)
    classes = y[np.argsort(flatten_distances)][:k]
    neighbors[0:k] = neighbors

    weights = np.zeros(k, dtype=np.float32)
    for i in range(k):
        weights[i] = 1 / LA.norm(x - neighbors[i], 2)

    for n_i in range(k):
        votes[classes[n_i]] += weights[n_i]

    return np.argmax(votes)

if __name__ == '__main__':
    # K = sys.argv[1]
    # D = sys.argv[2]
    # N = sys.argv[3]
    K = int(sys.argv[1])
    D = int(sys.argv[2])
    N = int(sys.argv[3])
    PATH_TO_DATA_DIR = sys.argv[4]

    description_label_bytes = 8
    description_images_bytes = 16

    # We should be getting this from header
    rows_N = 28
    columns_N = 28

    if not path.exists(os.path.join(PATH_TO_DATA_DIR, 'train-labels.idx1-ubyte')):
        print("File train-labels.idx1-ubyte does not exists at PATH_TO_DATA_DIR")

    if not path.exists(os.path.join(PATH_TO_DATA_DIR, 'train-images.idx3-ubyte')):
        print("File train-images.idx3-ubyte does not exists at PATH_TO_DATA_DIR")   

    with open(os.path.join(PATH_TO_DATA_DIR, 'train-labels.idx1-ubyte'), 'r') as file_in:
        size_bytes = os.fstat(file_in.fileno()).st_size

        m = mmap.mmap(file_in.fileno(), length=(IMAGE_SCOPE_SIZE) + description_label_bytes, access=mmap.ACCESS_READ)
        label_arr = np.frombuffer(m, np.uint8, offset=description_label_bytes)

    with open(os.path.join(PATH_TO_DATA_DIR, 'train-images.idx3-ubyte'), 'r') as file_in:
        size_bytes = os.fstat(file_in.fileno()).st_size

        m = mmap.mmap(file_in.fileno(), length=(IMAGE_SCOPE_SIZE * rows_N * columns_N) + description_images_bytes, access=mmap.ACCESS_READ)
        images_arr = np.frombuffer(m, np.ubyte, offset=description_images_bytes)

    # Section 1
    columnized_images = np.reshape(images_arr, (IMAGE_SCOPE_SIZE, 784))

    # Hint
    # print(np.mean(columnized_images[0]))

    training_set_x = columnized_images[N : IMAGE_SCOPE_SIZE, :]
    training_set_y = label_arr[N : IMAGE_SCOPE_SIZE]

    test_set_x = columnized_images[0 : N, :]
    test_set_y = label_arr[0 : N]

    # Test same image
    # plt.figure(figsize=(8,4))
    # f = test_set_x[0].reshape(28, 28)
    # s = training_set_x[0].reshape(28, 28)
    # w = columnized_images[0].reshape(28, 28)
    # u = columnized_images[N].reshape(28, 28)

    # ax0 = plt.subplot2grid((4, 2), (0, 0))
    # ax0.imshow(f, cmap='gray')
    # ax1 = plt.subplot2grid((4, 2), (1, 0))
    # ax1.imshow(s, cmap='gray')

    # ax2 = plt.subplot2grid((4, 2), (2, 0))
    # ax2.imshow(w, cmap='gray')
    # ax3 = plt.subplot2grid((4, 2), (3, 0))
    # ax3.imshow(u, cmap='gray')

    # plt.show()
    # for num_component in range(0, 784):

    # Section 2
    pca = PCA(n_components = D, svd_solver='full')
    transformed_training_set_x = pca.fit_transform(training_set_x)
    transformed_test_set_x = pca.transform(test_set_x)

    # Hint
    # print(transformed_test_set_x[0])

    # Section 3
    ypred = []

    for x in transformed_test_set_x:
        ypred.append(KNN(K, transformed_training_set_x, training_set_y, x))

    output_f = open("2211798052.txt", 'w')
    for index_pred in range(N):
        output_f.write(str(ypred[index_pred]))
        output_f.write(" ")
        output_f.write(str(test_set_y[index_pred]))
        output_f.write('\n') 
    output_f.close()

