from scipy.optimize import linear_sum_assignment
from scipy.sparse import csr_matrix
import networkx as nx


def hungarian_algorithm(grad):
    """
    https://docs.scipy.org/doc/scipy-0.18.1/reference/generated/scipy.optimize.linear_sum_assignment.html
    """
    row_ind, col_ind = linear_sum_assignment(grad)
    shape = len(row_ind)
    data = [1 for _ in range(shape)]
    return csr_matrix((data, (row_ind, col_ind)), shape=(shape, shape)).toarray()


def is_dag(B):
    """Check whether B corresponds to a DAG.

    Args:
        B (numpy.ndarray): [d, d] binary or weighted matrix.
    """
    return nx.is_directed_acyclic_graph(nx.DiGraph(B))
