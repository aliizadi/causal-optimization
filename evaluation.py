import numpy as np
from utils import is_dag


def count_accuracy(B_bin_true, B_bin_est, check_input=False):
    """Compute various accuracy metrics for B_bin_est.

    true positive = predicted association exists in condition in correct direction.
    reverse = predicted association exists in condition in opposite direction.
    false positive = predicted association does not exist in condition.

    Args:
        B_bin_true (np.ndarray): [d, d] binary adjacency matrix of ground truth. Consists of {0, 1}.
        B_bin_est (np.ndarray): [d, d] estimated binary matrix. Consists of {0, 1, -1},
            where -1 indicates undirected edge in CPDAG.

    Returns:
        fdr: (reverse + false positive) / prediction positive.
        tpr: (true positive) / condition positive.
        fpr: (reverse + false positive) / condition negative.
        shd: undirected extra + undirected missing + reverse.
        pred_size: prediction positive.

    Code modified from:
        https://github.com/xunzheng/notears/blob/master/notears/utils.py
    """
    if check_input:
        if (B_bin_est == -1).any():  # CPDAG
            if not ((B_bin_est == 0) | (B_bin_est == 1) | (B_bin_est == -1)).all():
                raise ValueError("B_bin_est should take value in {0, 1, -1}.")
            if ((B_bin_est == -1) & (B_bin_est.T == -1)).any():
                raise ValueError("Undirected edge should only appear once.")
        else:  # dag
            if not ((B_bin_est == 0) | (B_bin_est == 1)).all():
                raise ValueError("B_bin_est should take value in {0, 1}.")
            if not is_dag(B_bin_est):
                raise ValueError("B_bin_est should be a DAG.")
    d = B_bin_true.shape[0]
    # linear index of nonzeros
    pred_und = np.flatnonzero(B_bin_est == -1)
    pred = np.flatnonzero(B_bin_est == 1)
    cond = np.flatnonzero(B_bin_true)
    cond_reversed = np.flatnonzero(B_bin_true.T)
    cond_skeleton = np.concatenate([cond, cond_reversed])
    # true pos
    true_pos = np.intersect1d(pred, cond, assume_unique=True)
    # treat undirected edge favorably
    true_pos_und = np.intersect1d(pred_und, cond_skeleton, assume_unique=True)
    true_pos = np.concatenate([true_pos, true_pos_und])
    # false pos
    false_pos = np.setdiff1d(pred, cond_skeleton, assume_unique=True)
    false_pos_und = np.setdiff1d(pred_und, cond_skeleton, assume_unique=True)
    false_pos = np.concatenate([false_pos, false_pos_und])
    # reverse
    extra = np.setdiff1d(pred, cond, assume_unique=True)
    reverse = np.intersect1d(extra, cond_reversed, assume_unique=True)
    # compute ratio
    pred_size = len(pred) + len(pred_und)
    cond_neg_size = 0.5 * d * (d - 1) - len(cond)
    fdr = float(len(reverse) + len(false_pos)) / max(pred_size, 1)
    tpr = float(len(true_pos)) / max(len(cond), 1)
    fpr = float(len(reverse) + len(false_pos)) / max(cond_neg_size, 1)
    # structural hamming distance
    pred_lower = np.flatnonzero(np.tril(B_bin_est + B_bin_est.T))
    cond_lower = np.flatnonzero(np.tril(B_bin_true + B_bin_true.T))
    extra_lower = np.setdiff1d(pred_lower, cond_lower, assume_unique=True)
    missing_lower = np.setdiff1d(cond_lower, pred_lower, assume_unique=True)
    shd = len(extra_lower) + len(missing_lower) + len(reverse)
    return {'fdr': fdr, 'tpr': tpr, 'fpr': fpr, 'shd': shd, 'pred_size': pred_size}
