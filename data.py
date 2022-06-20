import numpy as np

seed = 0
np.random.seed(seed)

causal_func_dict = {'linear': lambda x, n: 2 * x + n,
                    'non-linear': lambda x, n: x + (.5) * x * x * x + (n),
                    # 'nueralnet_l1': lambda x, n: sigmoid(sigmoid(np.random.normal(loc=1) * x) + n),
                    # 'mnm': lambda x, n: sigmoid(np.random.normal(loc=1) * x) + .5 * x ** 2
                    #                     + sigmoid(np.random.normal(loc=1) * x) * n
                    }


def generate_bivariate_data(n_samples=100, direction='xy', causal_func='non-linear', noise_dist='gaussian'):
    if noise_dist == 'laplace':
        N = np.random.laplace(loc=0, scale=1. / np.sqrt(2), size=(n_samples, 2))
    elif noise_dist == 'gaussian':
        N = np.random.normal(loc=0, scale=1., size=(n_samples, 2))
    elif noise_dist == 'cauchy':
        N = np.random.standard_cauchy(size=(n_samples, 2))
    elif noise_dist == 'student':
        N = np.random.standard_t(df=5, size=(n_samples, 2))
    else:
        raise ValueError(noise_dist)

    N_laplace = np.random.laplace(loc=0, scale=1. / np.sqrt(2), size=(n_samples, 1))
    N_gaussian = np.random.normal(loc=0, scale=1., size=(n_samples, 1))

    X = np.zeros((n_samples, 2))
    X[:, 0] = N_gaussian[:, 0]
    X[:, 1] = causal_func_dict[causal_func](X[:, 0], N_laplace[:, 0])

    if direction == 'yx':
        X = X[:, [1, 0]]
        print(f'direction: {direction}')
    elif direction == 'xy':
        print(f'direction: {direction}')
        pass

    return X


def generate_multivariate_data(n_samples=100, direction='xy', causal_func='non-linear', noise_dist='gaussian'):
    dim = 4
    if noise_dist == 'laplace':
        N = np.random.laplace(loc=0, scale=1. / np.sqrt(2), size=(n_samples, dim))
    elif noise_dist == 'gaussian':
        N = np.random.normal(loc=0, scale=1., size=(n_samples, dim))
    elif noise_dist == 'cauchy':
        N = np.random.standard_cauchy(size=(n_samples, dim))
    elif noise_dist == 'student':
        N = np.random.standard_t(df=5, size=(n_samples, dim))
    else:
        raise ValueError(noise_dist)

    X = np.zeros((n_samples, dim))
    X[:, 0] = N[:, 0]
    X[:, 1] = causal_func_dict[causal_func](X[:, 0], N[:, 1])
    X[:, 2] = causal_func_dict[causal_func](X[:, 0], N[:, 2])
    X[:, 3] = causal_func_dict[causal_func](X[:, 1] + X[:, 2], N[:, 3])

    # new_permutation = np.random.permutation(X.shape[1])
    #
    # adjacency_matrix = np.zeros((dim, dim))
    # for i, permutation in enumerate(new_permutation.tolist()):
    #     adjacency_matrix[permutation, i] = 1
    #
    # print('*' * 50)
    # print('new_permutation: ', new_permutation)
    # print(adjacency_matrix)
    # print(adjacency_matrix[[0, 2, 1, 3], :])
    # print('*' * 50)
    #
    # X = X[:, new_permutation]

    return X
