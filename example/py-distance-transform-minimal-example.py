import py_distance_transform as dt
import numpy as np
import matplotlib.pyplot as plt


def distance_transform_example():
    mask = np.ones((51,100), dtype=float)*1e2
    mask[10,23] = 0
    mask[35,84] = 0

    distance_map = dt.distance_transform(mask,True)

    print("Max dist:", np.max(distance_map[:]))
    print(distance_map.dtype)

    plt.imshow(distance_map)
    plt.plot(23, 10, "mo")
    plt.plot(84, 35, "mo")
    plt.colorbar()
    plt.show()


if __name__ == "__main__":
    distance_transform_example()
