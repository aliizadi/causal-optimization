import numpy as np
import igraph as ig
import networkx as nx
import numpy as np

from utils import is_dag

seed = 0
np.random.seed(seed)


class SyntheticDataset:
    """Generate synthetic data.

    Key instance variables:
        X (numpy.ndarray): [n, d] data matrix.
        B (numpy.ndarray): [d, d] weighted adjacency matrix of DAG.
        B_bin (numpy.ndarray): [d, d] binary adjacency matrix of DAG.
    """

    def __init__(self, n, d, graph_type, degree, noise_type, B_scale=1, seed=1):
        """Initialize self.

        Args:
            n (int): Number of samples.
            d (int): Number of nodes.
            graph_type ('ER' or 'SF'): Type of graph.
            degree (int): Degree of graph.
            noise_type ('gaussian_ev', 'gaussian_nv', 'exponential', 'gumbel'): Type of noise.
            B_scale (float): Scaling factor for range of B.
            seed (int): Random seed. Default: 1.
        """
        self.n = n
        self.d = d
        self.graph_type = graph_type
        self.degree = degree
        self.noise_type = noise_type
        self.B_ranges = ((B_scale * -2.0, B_scale * -0.5),
                         (B_scale * 0.5, B_scale * 2.0))
        self.rs = np.random.RandomState(seed)  # Reproducibility

        self._setup()

    def _setup(self):
        """Generate B_bin, B and X."""
        self.B_bin = SyntheticDataset.simulate_random_dag(self.d, self.degree,
                                                          self.graph_type, self.rs)
        self.B = SyntheticDataset.simulate_weight(self.B_bin, self.B_ranges, self.rs)
        self.X = SyntheticDataset.simulate_linear_sem(self.B, self.n, self.noise_type, self.rs)
        assert is_dag(self.B)

    @staticmethod
    def simulate_er_dag(d, degree, rs=np.random.RandomState(1)):
        """Simulate ER DAG using NetworkX package.

        Args:
            d (int): Number of nodes.
            degree (int): Degree of graph.
            rs (numpy.random.RandomState): Random number generator.
                Default: np.random.RandomState(1).

        Returns:
            numpy.ndarray: [d, d] binary adjacency matrix of DAG.
        """

        def _get_acyclic_graph(B_und):
            return np.tril(B_und, k=-1)

        def _graph_to_adjmat(G):
            return nx.to_numpy_matrix(G)

        p = float(degree) / (d - 1)
        G_und = nx.generators.erdos_renyi_graph(n=d, p=p, seed=rs)
        B_und_bin = _graph_to_adjmat(G_und)  # Undirected
        B_bin = _get_acyclic_graph(B_und_bin)
        return B_bin

    @staticmethod
    def simulate_sf_dag(d, degree):
        """Simulate ER DAG using igraph package.

        Args:
            d (int): Number of nodes.
            degree (int): Degree of graph.

        Returns:
            numpy.ndarray: [d, d] binary adjacency matrix of DAG.
        """

        def _graph_to_adjmat(G):
            return np.array(G.get_adjacency().data)

        m = int(round(degree / 2))
        # igraph does not allow passing RandomState object
        G = ig.Graph.Barabasi(n=d, m=m, directed=True)
        B_bin = np.array(G.get_adjacency().data)
        return B_bin

    @staticmethod
    def simulate_random_dag(d, degree, graph_type, rs=np.random.RandomState(1)):
        """Simulate random DAG.

        Args:
            d (int): Number of nodes.
            degree (int): Degree of graph.
            graph_type ('ER' or 'SF'): Type of graph.
            rs (numpy.random.RandomState): Random number generator.
                Default: np.random.RandomState(1).

        Returns:
            numpy.ndarray: [d, d] binary adjacency matrix of DAG.
        """

        def _random_permutation(B_bin):
            # np.random.permutation permutes first axis only
            P = rs.permutation(np.eye(B_bin.shape[0]))
            return P.T @ B_bin @ P

        if graph_type == 'ER':
            B_bin = SyntheticDataset.simulate_er_dag(d, degree, rs)
        elif graph_type == 'SF':
            B_bin = SyntheticDataset.simulate_sf_dag(d, degree)
        else:
            raise ValueError("Unknown graph type.")
        return _random_permutation(B_bin)

    @staticmethod
    def simulate_weight(B_bin, B_ranges, rs=np.random.RandomState(1)):
        """Simulate the weights of B_bin.

        Args:
            B_bin (numpy.ndarray): [d, d] binary adjacency matrix of DAG.
            B_ranges (tuple): Disjoint weight ranges.
            rs (numpy.random.RandomState): Random number generator.
                Default: np.random.RandomState(1).

        Returns:
            numpy.ndarray: [d, d] weighted adjacency matrix of DAG.
        """
        B = np.zeros(B_bin.shape)
        S = rs.randint(len(B_ranges), size=B.shape)  # Which range
        for i, (low, high) in enumerate(B_ranges):
            U = rs.uniform(low=low, high=high, size=B.shape)
            B += B_bin * (S == i) * U
        return B

    @staticmethod
    def simulate_linear_sem(B, n, noise_type, rs=np.random.RandomState(1)):
        """Simulate samples from linear SEM with specified type of noise.

        Args:
            B (numpy.ndarray): [d, d] weighted adjacency matrix of DAG.
            n (int): Number of samples.
            noise_type ('gaussian_ev', 'gaussian_nv', 'exponential', 'gumbel'): Type of noise.
            rs (numpy.random.RandomState): Random number generator.
                Default: np.random.RandomState(1).

        Returns:
            numpy.ndarray: [n, d] data matrix.
        """

        def _simulate_single_equation(X, B_i):
            """Simulate samples from linear SEM for the i-th node.

            Args:
                X (numpy.ndarray): [n, number of parents] data matrix.
                B_i (numpy.ndarray): [d,] weighted vector for the i-th node.

            Returns:
                numpy.ndarray: [n,] data matrix.
            """
            if noise_type == 'gaussian_ev':
                # Gaussian noise with equal variances
                N_i = rs.normal(scale=1.0, size=n)
            elif noise_type == 'gaussian_nv':
                # Gaussian noise with non-equal variances
                scale = rs.uniform(low=1.0, high=2.0)
                N_i = rs.normal(scale=scale, size=n)
            elif noise_type == 'exponential':
                # Exponential noise
                N_i = rs.exponential(scale=1.0, size=n)
            elif noise_type == 'gumbel':
                # Gumbel noise
                N_i = rs.gumbel(scale=1.0, size=n)
            else:
                raise ValueError("Unknown noise type.")
            return X @ B_i + N_i

        d = B.shape[0]
        X = np.zeros([n, d])
        G = nx.DiGraph(B)
        ordered_vertices = list(nx.topological_sort(G))
        assert len(ordered_vertices) == d
        for i in ordered_vertices:
            parents = list(G.predecessors(i))
            X[:, i] = _simulate_single_equation(X[:, parents], B[parents, i])

        return X


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

    new_permutation = np.random.permutation(X.shape[1])

    adjacency_matrix = np.zeros((dim, dim))
    for i, permutation in enumerate(new_permutation.tolist()):
        adjacency_matrix[permutation, i] = 1

    print('*' * 50)
    print('new_permutation: ', new_permutation)
    print(adjacency_matrix)
    print(adjacency_matrix[[0, 2, 1, 3], :])
    print('*' * 50)

    X = X[:, new_permutation]

    return X
