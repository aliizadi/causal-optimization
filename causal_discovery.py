import numpy as np
from data import generate_multivariate_data
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from utils import hungarian_algorithm
import warnings
from data import SyntheticDataset
from evaluation import count_accuracy
from sklearn import linear_model
import cvxopt as cvx
from cvxopt import matrix
from ActiveSet import ActiveSet
from scipy.optimize import minimize
import networkx as nx
from cdt.metrics import SHD_CPDAG
from collections import defaultdict
import matlab.engine

warnings.filterwarnings("ignore")

seed = 5
np.random.seed(seed)
np.set_printoptions(suppress=True)


def get_parameters():
    return {'n_samples': 50000,
            'n_dimensions': 10,
            'graph_type': 'ER',  # in {'ER', 'SF'}
            'epochs': 1500,
            'convex_concave_epochs': 500,
            'test_size': 0.3,
            'print_epoch': 1,
            'direction': 'yx',
            'causal_func': 'linear',  # in {'linear', 'non-linear'}
            'noise': 'gaussian_nv',
            # in  {'gaussian', laplace, 'cauchy', 'student'} or {'exponential', 'gumbel', 'gaussian_ev', 'gaussian_nv}
            'dataset_type': generate_multivariate_data,  # in {generate_bivariate_data, generate_multivariate_data}
            'main_algorithm_convergence': 1e-15,
            'convex_concave_algorithm_convergence': 1e-6,
            'alpha': 0.1,
            'initial_mu': 0.1,
            'mu_increase': 0.01
            }


parameters = get_parameters()

eng = matlab.engine.start_matlab()


class QP(ActiveSet):

    def _calc_Hessians(self):
        self.H = self.A
        self.h = - self.b

    def _calc_objective(self, x):
        return x.T @ self.A @ x + self.b.T @ x

    def __call__(self, A, b, Ce=[], de=[], Ci=[], di=[], cu=[], cl=[], x0=[]):
        return self.run(A=A, b=b, Ce=Ce, de=de, Ci=Ci, di=di,
                        cu=cu, cl=cl, x0=x0)


class GraphMatching:
    def __init__(self, P, A_G, A_H):
        self.dim = P.shape[0]
        self.A_G, self.A_H = A_G, A_H
        self.P = P

    def optimize(self, A_G, A_H, mu, k):
        self.A_G, self.A_H = A_G, A_H
        losses = []
        for i in range(parameters['convex_concave_epochs']):
            self.P = self._qp_solver(mu)
            #     # print(np.round(self.P, 3))
            losses.append(self._loss(mu))
            # print(losses[-1])
            #
            if i > 1 and (losses[-2] - losses[-1]) > 0 and losses[-2] - losses[-1] < parameters[
                'convex_concave_algorithm_convergence']:
                #     # if i > 1 and losses[-2] - losses[-1] < parameters['convex_concave_algorithm_convergence']:
                break
        #
        print('inner algorithm converged at i: ', i)
        # self.P = self._qp_solver(mu)

        return self.P

    def _loss(self, mu):
        return np.trace(- self.P.T @ self.A_G.T @ self.P @ self.A_H) - mu * np.trace(self.P.T @ self.P)

    # def _f_mu(self, mu, P):
    #     A_G = self.A_G
    #     A_H = self.A_H
    #     H = - P.T @ A_G @ P @ A_H - mu * P.T @ P
    #     return np.trace(H)

    # def _grad_f_mu(self, mu, P):
    #     A_G = self.A_G
    #     A_H = self.A_H
    #     # H = P.T @ A_G @ P @ A_H + mu * P.T @ P
    #     grad_H = - 2 * A_G @ P @ A_H - 2 * mu * P
    #     return grad_H

    def _qp_solver(self, mu):
        n = self.dim * self.dim
        d = self.dim

        A_rows = np.zeros((d, n))
        for i in range(d):
            for j in range(i * d, (i + 1) * d):
                A_rows[i, j] = 1

        A_cols = np.zeros((d, n))
        for i in range(d):
            for j in range(d):
                A_cols[i, j * d + i] = 1

        I = np.eye(d)

        # Q = - np.kron(self.A_H.T, self.A_G.T) - np.eye(n) * 1e-20
        # Q = - np.kron(self.A_H.T, self.A_G.T) - mu * np.kron(I, I)
        Q = - np.kron(self.A_H.T, self.A_G.T)

        Q = (Q + Q.T) * 0.5
        # try:
        # assert np.all(np.linalg.eigvals(Q) >= 0)

        # except:
        #     print('ok')

        q = np.reshape(- mu * self.P, (n,))
        # q = np.zeros(n)

        G = - np.eye(n)
        h = np.zeros(n)

        A = np.vstack((A_rows, A_cols))
        b = np.ones(2 * d)

        # self.P = self._active_set(Q, q, A, b, G, h, self.P, n, d)
        # self.P = self._cvxpy(Q, q, A, b, G, h, self.P, n, d)
        # self.P = self._scipy(Q, q, A, b, G, h, self.P, n, d, method='SLSQP')

        self.P = self.matlab(Q, q, A, b, G, h, self.P, n, d)

        return self.P

    def _active_set(self, Q, q, A, b, G, h, P, n, d):
        Q = np.reshape(Q, (n, n))
        q = np.reshape(q, (n, 1))

        G = np.reshape(G, (n, n))
        h = np.reshape(h, (n, 1))

        A = np.reshape(A[:-1, :], (2 * d - 1, n))
        b = np.reshape(b[:-1], (2 * d - 1, 1))

        P = self._sinkhorn(self.P)
        P = np.reshape(P, (n, 1))

        opt = QP()

        x, var, nit = opt(Q, q, Ce=A, de=b, Ci=G, di=h, x0=P)
        P = np.array(x).reshape(self.dim, self.dim, order='F')

        return P

    def matlab(self, Q, q, A, b, G, h, P, n, d):

        P = np.reshape(P, (n, 1))

        Q = matlab.double(Q.tolist())
        q = matlab.double(q.tolist())
        G = matlab.double(G.tolist())
        h = matlab.double(h.tolist())
        A = matlab.double(A.tolist())
        b = matlab.double(b.tolist())
        P = matlab.double(P.tolist())

        options = eng.optimoptions('quadprog', 'Algorithm', 'active-set', 'Display', 'off')
        # ws = eng.optimwarmstart(P, options)

        x = eng.quadprog(Q, q, G, h, A, b, matlab.double([]), matlab.double([]), P, options)
        # x = eng.quadprog(Q, q, matlab.double([]), matlab.double([]), A, b, matlab.double([0 for _ in range(n)]),
        #                  matlab.double([1 for _ in range(n)]), P, options)
        #

        P = np.array(x).reshape(self.dim, self.dim, order='F')

        return P

    def _cvxpy(self, Q, q, A, b, G, h, P, n, d):
        Q = matrix(Q, (n, n))
        q = matrix(q, (n, 1))

        G = matrix(G, (n, n))
        h = matrix(h, (n, 1))

        A = matrix(A[:-1, :], (2 * d - 1, n))
        b = matrix(b[:-1], (2 * d - 1, 1))

        P = matrix(P, (n, 1))

        cvx.solvers.options['show_progress'] = False

        x = cvx.solvers.qp(Q, q, G, h, A, b, initvals=P)['x']

        P = np.array(x).reshape(self.dim, self.dim, order='F')

        return P

    def _scipy(self, Q, q, A, b, G, h, P, n, d, method):

        # Q = np.reshape(Q, (n, n))
        # q = np.reshape(q, (n, 1))
        # G = np.reshape(G, (n, n))
        # h = np.reshape(h, (n, 1))
        # A = np.reshape(A[:-1, :], (2 * d - 1, n))
        # b = np.reshape(b[:-1], (2 * d - 1, 1))
        # P = np.reshape(P, (n, 1))

        def loss(x):
            # print(x)
            return 0.5 * x.T @ Q @ x + q.T @ x

        def jac(x):
            return x.T @ Q + q

        ineq_cons = {'type': 'ineq',
                     'fun': lambda x: h - G @ x,
                     'jac': lambda x: -G}

        eq_cons = {'type': 'eq',
                   'fun': lambda x: A @ x - b,
                   'jac': lambda x: A}

        # P = np.random.random((self.dim, self.dim))
        res_cons = minimize(loss, P, jac=jac, constraints=[eq_cons, ineq_cons],
                            method=method, options={'disp': False, 'maxiter': 1000})
        x = res_cons['x']

        P = np.array(x).reshape(self.dim, self.dim, order='F')

        return P

    def _sinkhorn(self, P):

        X = P
        S0 = np.exp(X)

        ones = np.ones((self.dim, 1))

        def T_C(X):
            return np.divide(X, ones @ ones.T @ X)

        def T_R(X):
            return np.divide(X, X @ ones @ ones.T)

        S = S0
        for i in range(300):
            S = T_C(T_R(S))

        return S

        # def _backtracking_line_search(self, x_k, d_k, grad_f_k, mu, initial_alpha=1.0, c1=10e-4, c2=0.99, rho=0.5):
    #     def armijo(alpha):
    #         # return f(x[0] + alpha * p[0], x[1] + alpha * p[1]) <= f(x[0], x[1]) + \
    #         #        c1 * alpha * np.dot(grad_f(x[0], x[1]), p)
    #
    #         return self._f_mu(mu, x_k + alpha * d_k) <= self._f_mu(mu, x_k) + c1 * alpha * np.sum(np.dot(grad_f_k, d_k))
    #
    #     alpha = initial_alpha
    #
    #     while not armijo(alpha):
    #         alpha = alpha * rho
    #
    #     return alpha


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

    def solve(self, alpha):
        solution, _, _, _ = np.linalg.lstsq(self.left_side, self.right_side, rcond=None)

        # clf = linear_model.Lasso(alpha=alpha)
        # clf.fit(self.left_side, self.right_side)
        # solution = clf.coef_

        A = np.zeros((self.dim, self.dim))
        for i in range(self.dim):
            for j in range(i):
                A[i, j] = solution[int((i - 1) * i / 2) + j]

        return A.T


class LinearCausalDiscovery:
    def __init__(self, train, validation, dag):
        self.train = train
        self.validation = validation
        self.dim = train.shape[1]
        self.dag = dag
        self.A = np.triu(np.random.random((self.dim, self.dim)), k=1)
        self.P = np.ones((self.dim, self.dim)) / self.dim
        # self.P = np.array([[0.3, 0.7],
        #                    [0.7, 0.3]])
        # self.P = np.random.random((self.dim, self.dim))

    def optimize(self):
        train_losses = []
        validation_losses = []
        permutation_values = []
        Ps = []
        As = []
        graph_matching = GraphMatching(self.P, self.A_G, self.A_H_train)

        # train_loss = self.loss(data_type='train')
        # validation_loss = self.loss(data_type='validation')
        # train_losses.append(train_loss)
        # validation_losses.append(validation_loss)

        alpha = 0

        mu = parameters['initial_mu']
        mu_increase = parameters['mu_increase']

        for epoch in range(parameters['epochs']):

            self.P = graph_matching.optimize(self.A_G, self.A_H_train, mu, epoch)
            Ps.append(self.P)

            least_square = LeastSquare(self.P, self.train)
            self.A = least_square.solve(alpha)
            As.append(self.A)

            train_loss = self.loss(data_type='train')
            validation_loss = self.loss(data_type='validation')

            permutation_value = self._permutation_value()
            permutation_values.append(permutation_value)

            if epoch > 1:
                alpha = parameters['alpha'] * permutation_value

                diff_permutation_value = permutation_values[-1] - permutation_values[-2]

                if diff_permutation_value < 0.01 or mu_increase < 1e-4:

                    # if mu_increase < 1e-4:
                    #     print('first condition 1')
                    #     mu_increase *= 32
                    #     mu /= 1 + mu_increase
                    #
                    # else:
                    #     mu *= 1 + mu_increase
                    #     print('first condition 2')

                    mu *= 1 + mu_increase

                elif diff_permutation_value > 0.01:
                    self.P = Ps[-2]
                    self.A = As[-2]
                    Ps = Ps[:-1]
                    As = As[:-1]
                    mu /= 1 + mu_increase
                    mu_increase *= 0.5
                    mu *= 1 + mu_increase
                    permutation_value = permutation_values[-2]
                    permutation_values = permutation_values[:-1]
                    print('second condition')

                    # if mu_diff_permutation_valueincrease < 1e-4:
                    #     mu *= (1 + mu_increase)

                if diff_permutation_value < 0.001:
                    mu_increase *= 2
                    print('third condition')

            if epoch % parameters['print_epoch'] == 0:
                print(
                    'epoch: {}, train loss: {}, validation loss: {}'.format(epoch, train_loss, validation_loss))
                print('permutation_value: ', permutation_value)
                if epoch > 1:
                    print('diff permutation_value: ', diff_permutation_value)
                print('mu: ', mu)
                print('mu_increase: ', mu_increase)
                print('alpha: ', alpha)

                print('-' * 100)
                # print('----------------- A:')
                # print(np.round(self.A, 2))
                # print('----------------- P:')
                # print(np.round(self.P, 2))
                # if epoch > 1 and abs(train_losses[-2] - train_losses[-1]):
                #     print('loss change: ', abs(train_losses[-2] - train_losses[-1]))
                # print('-' * 100)

            train_losses.append(train_loss)
            validation_losses.append(validation_loss)

            # if (epoch > 1 and abs(train_losses[-2] - train_losses[-1]) < parameters['main_algorithm_convergence']) or \
            #         epoch == parameters['epochs'] - 1:
            if epoch > 1 and permutation_value > 0.999:
                break

        print(f'algorithm converged at epoch: {epoch} with training loss: {train_loss}'
              f' validation loss: {validation_loss}')
        print('----------------- P:')
        print(np.abs(np.round(self.P, 2)))
        print('----------------- A:')
        print(np.round(self.A, 2))
        #
        # print('----------------- P.T @ A @ P:')
        # pred_dag = self.P.T @ self.A @ self.P
        pred_dag = np.round(self.P.T @ self.A @ self.P, 2)

        print('----------------- Pred Dag:')
        print(np.round(pred_dag, 2))

        self._topological_order_metric(self.dag, self.P)

        print('----------------- True Dag:')
        print(self.dag)

        print('*' * 20, ' evaluation ', '*' * 20)
        accuracies = count_accuracy(self.dag != 0, pred_dag != 0)
        shd = accuracies['shd']
        print(f'shd: {shd}')

        print(f'CPDAG shd: {SHD_CPDAG(self.dag != 0, pred_dag != 0)}')

        self._bfs(pred_dag)

        return self.A, self.P, train_losses, validation_losses, shd, permutation_values

    @property
    def A_G(self):
        return self.A.T + self.A - self.A @ self.A.T - np.eye(self.dim)

    @property
    def A_H_train(self):
        return (self.train.T @ self.train) / self.train.shape[0]

    @property
    def A_H_validation(self):
        return (self.validation.T @ self.validation) / self.validation.shape[0]

    def loss(self, data_type='train'):
        assert data_type in ['train', 'validation']
        A_H = self.A_H_train if data_type == 'train' else self.A_H_validation
        A_G = self.A_G

        return np.trace(- self.P.T @ A_G.T @ self.P @ A_H)

    def _permutation_value(self):
        return np.trace(self.P @ self.P.T) / self.dim

    def _bfs(self, pred_dag):

        result = []
        for i in range(self.dim):
            for j in range(self.dim):
                if np.round(self.P[i, j], 2) == 1.:
                    result.append(j)

        print(result)
        print()

        print('pred dag bfs')

        G = nx.from_numpy_matrix(pred_dag, create_using=nx.DiGraph)
        nx.draw(G, with_labels=True)
        # plt.title('pred')
        for i in result:
            print(dict(enumerate(nx.bfs_layers(G, [i]))))

        print('main dag bfs')

        plt.figure()
        G = nx.from_numpy_matrix(self.dag, create_using=nx.DiGraph)
        nx.draw(G, with_labels=True)
        # plt.title('true')
        for i in result:
            print(dict(enumerate(nx.bfs_layers(G, [i]))))

    def _topological_order_metric(self, main_dag, permutation):
        permutation = np.round(permutation, 2)
        topological_order = []
        for i in range(self.dim):
            for j in range(self.dim):
                if permutation[i, j] == 1.:
                    topological_order.append(j)

        adjacency_list = defaultdict(list)
        for i in range(main_dag.shape[0]):
            for j in range(main_dag.shape[1]):
                if main_dag[i, j] == 1:
                    adjacency_list[i].append(j)

        def dfs(data, path, paths):
            datum = path[-1]
            if datum in data:
                for val in data[datum]:
                    new_path = path + [val]
                    paths = dfs(data, new_path, paths)
            else:
                paths += [path]
            return paths

        def enumerate_paths(graph):
            nodes = list(graph.keys())
            all_paths = []
            for node in nodes:
                node_paths = dfs(graph, [node], [])
                all_paths += node_paths
            return all_paths

        directed_paths = enumerate_paths(adjacency_list)

        for directed_path in directed_paths:
            print(directed_path)

        wrong_order = {}
        for i in range(len(topological_order) - 1):
            x_current = topological_order[i]
            x_next = topological_order[i + 1]
            found_correct_order = False
            for directed_path in directed_paths:

                if x_current in directed_path and x_next in directed_path and directed_path.index(
                        x_current) < directed_path.index(x_next):
                    found_correct_order = True
                    break

            if not found_correct_order:
                wrong_order[x_current] = x_next

        print(wrong_order)


def get_train_validation(d=4, degree=4):
    # n, d = parameters['n_samples'], parameters['n_dimensions']
    n = parameters['n_samples']

    graph_type, degree = parameters['graph_type'], degree  # ER2 graph
    B_scale = 1.0
    noise_type = parameters['noise']
    synthetic_dataset = SyntheticDataset(n, d, graph_type, degree,
                                         noise_type, B_scale, seed=seed)

    # print('-' * 100)
    # print(synthetic_dataset.B)
    # print(synthetic_dataset.B_bin)
    # print('-' * 100)

    # dataset_type = parameters['dataset_type']
    # n_samples = parameters['n_samples']
    # causal_func = parameters['causal_func']
    # noise = parameters['noise']
    # direction = parameters['direction']
    #
    # noisy_dataset = dataset_type(n_samples, direction, causal_func, noise)
    # train, validation = train_test_split(noisy_dataset, test_size=0.3)

    train, validation = train_test_split(synthetic_dataset.X, test_size=0.3)
    return train, validation, synthetic_dataset.B_bin
    # return train, validation, None


def optimize():
    shds = []
    # dimensions = [2, 3, 4, 5, 10, 15, 20, 30, 50, 100]
    dimensions = [parameters['n_dimensions']]

    for d in dimensions:
        print('-' * 100)
        print(f'starting optimization for dimension: {d}')
        train, validation, dag = get_train_validation(d=d)

        linear_causal_discovery = LinearCausalDiscovery(train, validation, dag)

        A, P, train_loss, validation_loss, shd, tr_ppt = linear_causal_discovery.optimize()

        shds.append(shd)

    # plt.plot(dimensions, shds, linestyle='dashed', marker='o')
    # plt.xlabel('n_dimensions')
    # plt.ylabel('shd')
    # plt.title(f'shd of different dimensions, n_samples: {parameters["n_samples"]}')

    return train_loss, validation_loss, tr_ppt


def plot_losses(train_loss, validation_loss, tr_ppt):
    n_samples = parameters['n_samples']

    plt.figure()
    plt.plot(train_loss, label='train', c='blue')
    plt.plot(validation_loss, label='validation', c='black')

    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.title(f'loss - n_samples = {n_samples}')
    plt.legend()

    plt.figure()
    plt.plot(tr_ppt)
    plt.title(f'permutation: tr(P.T @ P)')


def run():
    train_loss, validation_loss, tr_ppt = optimize()
    plot_losses(train_loss, validation_loss, tr_ppt)


run()
eng.quit()

plt.show()
