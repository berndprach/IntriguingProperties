import numpy as np

from lipschitz.algorithms.pca import PCA


def test_pca_gives_correct_directions():
    data_matrix = np.array([[101, 1], [-101, 0], [99, -1], [-90, 0]])

    pca = PCA()
    pca.fit(data_matrix)

    target = np.array([[1, 0], [0, 1]])
    assert np.allclose(np.absolute(pca.w), target, atol=0.1)


def test_pca_gives_correct_directions_no_axis_aligned():
    data_matrix = np.array([[30, 10], [301, 101], [-150, -53], [-3, -1]])

    pca = PCA()
    pca.fit(data_matrix)

    target = np.array([[3, 1]]) / np.sqrt(10)
    assert (np.allclose(pca.components[0], target, atol=0.1)
            or np.allclose(pca.components[0], -target, atol=0.1))


def test_reconstruction():
    # data: ~ [20, -10] + x * [3, 1]
    data_matrix = np.array([[20, -10], [23, -9], [50, 1], [-11, -20]])

    pca = PCA()
    pca.fit(data_matrix)

    target = np.array([29, -7])
    x = target + 3 * np.array([-1, 3])
    x_hat = pca.reconstruct(x, [0])

    assert np.allclose(x_hat, target, rtol=0.1)
