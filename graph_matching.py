import numpy as np
from data import generate_bivariate_data, generate_multivariate_data
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from scipy.sparse import csgraph
from scipy.optimize import linear_sum_assignment
from scipy.sparse import csr_matrix
import cvxpy as cp

seed = 0
np.random.seed(seed)


def get_parameters():
    return {'n_samples': 5000,
            'epochs': 100,
            'test_size': 0.3,
            'print_epoch': 1,
            'direction': 'yx',
            'causal_func': 'linear',  # in {'linear', 'non-linear'}
            'noise': 'laplace',  # in  {'gaussian', laplace, 'cauchy', 'student'}
            'dataset_type': generate_multivariate_data,  # in {generate_bivariate_data, generate_multivariate_data}
            'lambda_A': 10e-1,  # sparsity coefficient for lower triangular adjacency matrix A
            'eta_lambda_A': 10e-1,
            'frank_wolfe_epochs': 1000,
            'frank_wolfe_convergence': 10e-6,
            'd_lambda_': 0.1,
            'path_algorithm_convergence': 10e-6,
            }


parameters = get_parameters()


class GraphMatching:
    def __init__(self, A_G, A_H, P):
        self.dim = P.shape[0]
        self.A_G = A_G
        self.A_H = A_H
        self.P = P

    def _convex_relaxation(self):
        graph_matching = self.A_G - self.P @ self.A_H @ self.P.T
        f_0 = np.linalg.norm(graph_matching, ord='fro') ** 2
        return f_0

    def _concave_relaxation(self):
        # L = D - A  : D is the degree matrix and L is the Laplacian matrix
        self.L_A_H, self.D_A_H = self._laplacian(self.A_H)
        self.L_A_G, self.D_A_G = self._laplacian(self.A_G)

        self.delta = np.zeros((self.dim, self.dim))
        for i in range(self.dim):
            for j in range(self.dim):
                self.delta[i, j] = (self.D_A_H[j, j] - self.D_A_G[i, i]) ** 2

        self.vec_P = self.P.flatten(order='F')  # vectorized column-major order

        f1 = -np.trace(self.delta @ self.P) - 2 * self.vec_P.T @ np.kron(self.L_A_H.T, self.L_A_G.T) @ self.vec_P
        return f1

    def _laplacian(self, matrix):
        L = csgraph.laplacian(matrix, normed=False)
        D = L + matrix
        return L, D

    def _path_algorithm(self):
        lambda_ = 0
        d_lambda_ = parameters['d_lambda_']
        self._f_lambda(lambda_)
        self.P = self._qp_solver()
        print('p: \n', np.round(self.P, 3))
        print('-' * 60)

        iteration = 0
        while lambda_ < 1:
            lambda_new = lambda_ + d_lambda_
            if np.abs(self._f_lambda(lambda_new) - self._f_lambda(lambda_)) <= parameters['path_algorithm_convergence']:
                lambda_ = lambda_new
            else:
                self.P = self._frank_wolfe(lambda_new)
                lambda_ = lambda_new

            iteration += 1

            # if iteration % 10 == 0:
            print(f'path algorithm iteration: {iteration}, lambda: {lambda_}')
            print('p: \n', np.round(self.P, 3))
            print('-' * 60)

        self.P = self._frank_wolfe(lambda_)

        return self.P

    def _f_lambda(self, lambda_):
        return (1 - lambda_) * self._convex_relaxation() + lambda_ * self._concave_relaxation()

    def _frank_wolfe(self, lambda_):
        """
        https://en.wikipedia.org/wiki/Frank%E2%80%93Wolfe_algorithm
        """
        for k in range(parameters['frank_wolfe_epochs']):
            # step 1: estimation of  the gradient
            grad = self._grad_f_lambda(lambda_)

            # step 2: resolution of the linear program
            # s = self._hungarian_algorithm(grad)
            s = self._lp_solver(grad.flatten(order='F'))

            d = s - self.P

            g = - np.sum(np.dot(grad, d))

            if g <= parameters['frank_wolfe_convergence']:
                print(f'wolfe condition satisfied at epoch: {k}')
                break

            # step 3: line search
            # alpha = 2 / (k + 2)
            X = self.A_G @ d - d @ self.A_H
            Y = self.A_G @ self.P - self.P @ self.A_H

            alpha = min(1, - np.sum(np.dot((X.T @ X), (X.T @ Y)) / np.sum(np.dot(X.T @ X, X.T @ X))))

            # step 4: update of p
            self.P = self.P + alpha * d

        # if k == parameters['frank_wolfe_epochs'] - 1:
        #     print(f'wolfe condition not satisfied at epoch: {k}')

        return self.P

    def _grad_f_lambda(self, lambda_):
        grad_f0 = 2 * (self.A_G @ self.A_G @ self.P -
                       self.A_G.T @ self.P @ self.A_H.T -
                       self.A_G @ self.P @ self.A_H +
                       self.P @ self.A_H @ self.A_H).flatten(order='F')

        grad_f1 = -self.delta.T.flatten(order='F') - 2 * (self.L_A_G.T @ self.P @ self.L_A_H).flatten(order='F')

        return ((1 - lambda_) * grad_f0 + lambda_ * grad_f1).reshape(self.dim, self.dim, order='f')

    def _hungarian_algorithm(self, grad):
        """
        https://docs.scipy.org/doc/scipy-0.18.1/reference/generated/scipy.optimize.linear_sum_assignment.html
        """
        row_ind, col_ind = linear_sum_assignment(grad)
        shape = len(row_ind)
        data = [1 for _ in range(shape)]
        return csr_matrix((data, (row_ind, col_ind)), shape=(shape, shape)).toarray()

    def _qp_solver(self):
        n = self.dim * self.dim
        d = self.dim

        Q = np.kron(self.A_H.T, self.A_G.T)

        assert np.all(np.linalg.eigvals(Q) <= 0)

        h = np.zeros(n)

        A_rows = np.zeros((d, n))
        for i in range(d):
            for j in range(i * d, (i + 1) * d):
                A_rows[i, j] = 1

        A_cols = np.zeros((d, n))
        for i in range(d):
            for j in range(d):
                A_cols[i, j * d + i] = 1

        A = np.vstack((A_rows, A_cols))

        b = np.ones(2 * d)

        x = cp.Variable(n)
        prob = cp.Problem(cp.Maximize((1 / 2) * cp.quad_form(x, Q)),
                          [x >= h,
                           A @ x == b])
        #
        prob.solve()
        self.P = x.value.reshape(self.dim, self.dim, order='F')

        # closed form based on paper:
        # B = np.vstack((A_rows, A_cols))
        # I = np.eye(d)
        # Q_inv = np.linalg.inv(np.kron(self.A_H, self.A_G))
        # U_AH, S_AH, _ = np.linalg.svd(self.A_H, full_matrices=True)
        # U_AG, S_AG, _ = np.linalg.svd(self.A_G, full_matrices=True)
        # S_AH = np.diag(S_AH)
        # S_AG = np.diag(S_AG)
        # first_term = np.kron(U_AH, U_AG)
        # second_term = np.linalg.inv(np.kron(I, S_AG) - np.kron(S_AH, I))
        # Q_inv = first_term @ second_term @ second_term @ first_term.T
        # vec_p = Q_inv @ B.T @ np.linalg.pinv(B @ Q_inv @ B.T) @ b
        # self.P = vec_p.reshape(self.dim, self.dim, order='F')

        return self.P

    def _lp_solver(self, grad):
        n = self.dim * self.dim
        d = self.dim

        h = np.zeros(n)

        A_rows = np.zeros((d, n))
        for i in range(d):
            for j in range(i * d, (i + 1) * d):
                A_rows[i, j] = 1

        A_cols = np.zeros((d, n))
        for i in range(d):
            for j in range(d):
                A_cols[i, j * d + i] = 1

        A = np.vstack((A_rows, A_cols))

        b = np.ones(2 * d)

        x = cp.Variable(n)
        prob = cp.Problem(cp.Minimize(grad.T @ x),
                          [x >= h,
                           A @ x == b])
        prob.solve()

        return x.value.reshape(self.dim, self.dim, order='F')

    def optimize(self, type='path0'):
        if type == 'path0':
            P = self._qp_solver()

        elif type == 'path':
            P = self._path_algorithm()
        return P

    def loss(self):
        A_G = self.A_G
        A_H = self.A_H
        return -np.trace(self.P.T @ A_G.T @ self.P @ A_H)


class LeastSquare:
    def __init__(self, P, train):
        self.P = P
        self.dim = train.shape[1]
        self.n_samples = train.shape[0]
        self.train = train.T
        self.train = self.P @ self.train

        self.right_side = self.train[1:, :].flatten(order='F')
        s = int((self.dim - 1) * self.dim / 2)
        self.left_side = np.zeros(((self.dim - 1) * self.n_samples, s))
        self._fill_left_side()

    def _fill_left_side(self):
        for i in range(self.n_samples):
            for j in range(1, self.dim):
                for k in range(int((j - 1) * j / 2), int((j - 1) * j / 2) + j):
                    self.left_side[i * (self.dim - 1) + j - 1, k] = self.train[k - int((j - 1) * j / 2), i]

    def solve(self):
        solution, _, _, _ = np.linalg.lstsq(self.left_side, self.right_side, rcond=None)

        A = np.zeros((self.dim, self.dim))
        for i in range(self.dim):
            for j in range(i):
                A[i, j] = solution[int((i - 1) * i / 2) + j]

        return A.T


class LinearCausalDiscovery:
    def __init__(self, train, validation):
        self.train = train
        self.validation = validation
        self.dim = train.shape[1]

        self.A = np.tril(np.random.random((self.dim, self.dim)), k=-1)
        self.P = np.eye(self.dim)
        # self.P = np.ones((self.dim, self.dim)) / self.dim
        # self.P = np.array([[0, 0, 0, 1],
        #                    [0, 0, 1, 0],
        #                    [0, 1, 0, 0],
        #                    [1, 0, 0, 0]])

        # self.P = np.array([[0, 0, 0, 1],
        #                    [0, 1, 0, 0],
        #                    [0, 0, 1, 0],
        #                    [1, 0, 0, 0]])

        self.lambda_A = parameters['lambda_A']
        self.eta_lambda_A = parameters['eta_lambda_A']

    def _hungarian_algorithm(self, grad):
        """
        https://docs.scipy.org/doc/scipy-0.18.1/reference/generated/scipy.optimize.linear_sum_assignment.html
        """
        row_ind, col_ind = linear_sum_assignment(grad)
        shape = len(row_ind)
        data = [1 for _ in range(shape)]
        return csr_matrix((data, (row_ind, col_ind)), shape=(shape, shape)).toarray()

    def optimize(self):
        train_losses = []
        validation_losses = []
        for epoch in range(parameters['epochs']):
            least_square = LeastSquare(self.P, self.train)
            graph_matching = GraphMatching(self.A_G, self.A_H_train, self.P)

            self.A = least_square.solve()
            self.P = graph_matching.optimize()
            # self.P = self._hungarian_algorithm(-self.P)

            train_loss = self.loss(data_type='train')
            validation_loss = self.loss(data_type='validation')
            print('----------------- A:')
            print(np.round(self.A, 2))
            print('----------------- P:')
            print(np.round(self.P, 2))
            if epoch % parameters['print_epoch'] == 0:
                print('epoch: {}, train loss: {}, validation loss: {}'.format(epoch, train_loss, validation_loss))

            train_losses.append(train_loss)
            validation_losses.append(validation_loss)

        return self.A, self.P, train_losses, validation_losses

    @property
    def A_G(self):
        return self.A.T + self.A - self.A @ self.A.T - np.eye(self.dim)

    @property
    def A_H_train(self):
        return np.cov(self.train.T)

    @property
    def A_H_validation(self):
        return np.cov(self.validation.T)

    def loss(self, data_type='train'):
        A_G = self.A_G
        A_H = self.A_H_train if data_type == 'train' else self.A_H_validation
        return -np.trace(self.P.T @ A_G.T @ self.P @ A_H)


def get_train_validation():
    dataset_type = parameters['dataset_type']
    n_samples = parameters['n_samples']
    causal_func = parameters['causal_func']
    noise = parameters['noise']
    direction = parameters['direction']

    noisy_dataset = dataset_type(n_samples, direction, causal_func, noise)
    train, validation = train_test_split(noisy_dataset, test_size=0.3)
    return train, validation


def optimize():
    train, validation = get_train_validation()

    linear_causal_discovery = LinearCausalDiscovery(train, validation)

    A, P, train_loss, validation_loss = linear_causal_discovery.optimize()

    print('----------------- A:')
    print(np.round(A, 2))
    print('----------------- P:')
    print(np.round(P, 2))
    print('----------------- P.T @ A @ P:')
    print(np.round(P.T @ A @ P, 2))

    return train_loss, validation_loss


def plot_losses(train_loss, validation_loss):
    n_samples = parameters['n_samples']
    plt.figure()
    plt.plot(train_loss, label='train')
    # plt.plot(validation_loss, label='validation')

    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.title(f'loss - \n n_samples = {n_samples}')
    plt.legend()


def run():
    train_loss, validation_loss = optimize()
    plot_losses(train_loss, validation_loss)

    return train_loss, validation_loss


run()

plt.show()
