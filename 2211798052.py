import sys
import mmap
import numpy as np
import os
import matplotlib.pyplot as plt

IMAGE_SCOPE_SIZE = 800


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

    # TODO: Remove this
    rows_N = 28
    columns_N = 28

    label_arr = None

    with open(os.path.join(PATH_TO_DATA_DIR, 'train-labels.idx1-ubyte'), 'r') as file_in:
        size_bytes = os.fstat(file_in.fileno()).st_size

        # TODO VERY IMPORTANT Try to get headers instead of hardcoding

        m = mmap.mmap(file_in.fileno(), length=(IMAGE_SCOPE_SIZE) + description_label_bytes, access=mmap.ACCESS_READ)
        label_arr = np.frombuffer(m, np.uint8, offset=description_label_bytes)

    with open(os.path.join(PATH_TO_DATA_DIR, 'train-images.idx3-ubyte'), 'r') as file_in:
        size_bytes = os.fstat(file_in.fileno()).st_size

        # TODO VERY IMPORTANT Try to get headers instead of hardcoding

        m = mmap.mmap(file_in.fileno(), length=(IMAGE_SCOPE_SIZE * rows_N * columns_N) + description_images_bytes, access=mmap.ACCESS_READ)
        images_arr = np.frombuffer(m, np.ubyte, offset=description_images_bytes)

    columnized_images = np.reshape(images_arr, (IMAGE_SCOPE_SIZE, 784))

    # Hint
    # print(np.mean(columnized_images[0]))

    test_set_x = columnized_images[0:N, :]
    training_set_x = columnized_images[N:IMAGE_SCOPE_SIZE, :]

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

    print("end of main function")