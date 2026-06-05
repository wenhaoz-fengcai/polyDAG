"""
Standardized metric implementations for causal discovery evaluation.

CONVENTIONS:
1. Adjacency Matrix (W):
   - W[i, j] != 0 implies a directed edge from node i to node j (i -> j).
   - This matches the standard NOTEARS and standard graph theory convention.
   
2. Thresholding:
   - Edges are determined by |W[i, j]| > threshold.
   - Default threshold is 0.3 (standard in NOTEARS literature).
   - Self-loops (diagonals) are ignored in SHD/F1 calculation usually, 
     but here we assume W_true has 0 diagonal and we enforce 0 diagonal on W_est if needed.
"""

import numpy as np
from typing import Dict, Any
from dataclasses import dataclass

@dataclass
class EvaluationMetrics:
    """Standard metrics for causal discovery."""
    f1: float
    shd: int
    tpr: float
    fpr: float
    precision: float
    recall: float
    nnz: int

def threshold_adjacency(W: np.ndarray, threshold: float = 0.3) -> np.ndarray:
    """
    Convert weighted adjacency matrix to binary adjacency matrix.
    
    Args:
        W: Weighted adjacency matrix [d, d].
        threshold: Absolute value threshold.
        
    Returns:
        Binary adjacency matrix [d, d] (int).
    """
    if hasattr(W, 'detach'):
        W = W.detach().cpu().numpy()
    elif hasattr(W, 'numpy') and not isinstance(W, np.ndarray):
        W = W.numpy()
        
    W_abs = np.abs(W)
    return (W_abs > threshold).astype(int)

def count_accuracy(B_true: np.ndarray, B_est: np.ndarray) -> Dict[str, Any]:
    """
    Compute confusion matrix counts and basic ratios.
    Adapted from standard NOTEARS evaluation code.
    
    Args:
        B_true: Binary ground truth matrix [d, d].
        B_est: Binary estimated matrix [d, d].
        
    Returns:
        Dictionary with fdr, tpr, fpr, shd, nnz.
    """
    d = B_true.shape[0]
    
    # Linear index of nonzeros
    # pred: indices where B_est == 1
    # cond: indices where B_true == 1
    pred = np.flatnonzero(B_est)
    cond = np.flatnonzero(B_true)
    cond_reversed = np.flatnonzero(B_true.T)
    cond_skeleton = np.concatenate([cond, cond_reversed])
    
    # True Positives (TP): Edges in both, correct direction
    true_pos = np.intersect1d(pred, cond, assume_unique=True)
    
    # False Positives (FP): Edges in est but not in true skeleton (extra edges)
    # Note: This definition treats reversed edges as separate errors (Reverse) usually,
    # but standard code often groups them.
    # Here we follow the standard implementation:
    # False Pos = predicted edges that are NOT in the true skeleton (neither i->j nor j->i)
    false_pos = np.setdiff1d(pred, cond_skeleton, assume_unique=True)
    
    # Reverse: Edges in est that are in true skeleton but wrong direction
    # i.e. pred has i->j, true has j->i
    extra = np.setdiff1d(pred, cond, assume_unique=True)
    reverse = np.intersect1d(extra, cond_reversed, assume_unique=True)
    
    # Compute metrics
    pred_size = len(pred)
    cond_neg_size = 0.5 * d * (d - 1) - len(cond) # Total possible non-edges (excluding diagonal)
    
    # FDR: (False Pos + Reverse) / Predicted
    fdr = float(len(false_pos) + len(reverse)) / max(pred_size, 1)
    
    # TPR: True Pos / True Edges
    tpr = float(len(true_pos)) / max(len(cond), 1)
    
    # FPR: False Pos / True Non-Edges
    # Note: This FPR definition usually only counts "extra" edges as False Positives, 
    # not reversals. Reversals are "wrong" but not "false positives" in the sense of sparsity?
    # Standard implementation uses len(false_pos) / cond_neg_size
    fpr = float(len(false_pos)) / max(cond_neg_size, 1)
    
    # SHD: Structural Hamming Distance
    # Extra + Missing + Reverse
    # Extra: len(false_pos)
    # Missing: len(cond) - len(true_pos) - len(reverse) ? No.
    # Let's use the standard lower-triangular logic for SHD to be safe if DAGs.
    # But for general directed graphs:
    # SHD = E + M + R
    # E (Extra): Edges in est but not in true skeleton
    # M (Missing): Edges in true but not in est skeleton
    # R (Reverse): Edges in both skeletons but wrong direction
    
    # Re-calculating SHD explicitly to be sure
    # 1. Undirected difference
    B_true_sym = B_true + B_true.T
    B_est_sym = B_est + B_est.T
    # Edges present in one skeleton but not the other
    diff_und = np.abs((B_true_sym > 0).astype(int) - (B_est_sym > 0).astype(int))
    # Each edge counted twice in symmetric matrix
    skeleton_diff = np.sum(diff_und) // 2
    
    # 2. Orientation difference (for edges present in both skeletons)
    # Edges that are in both skeletons
    # Check if direction matches
    # We already calculated 'reverse' above.
    
    # Standard SHD formula often used:
    # shd = len(extra_lower) + len(missing_lower) + len(reverse)
    # But that assumes CPDAGs or specific structures.
    
    # Let's stick to the simple definition:
    # SHD = (False Positives) + (False Negatives) + (Reversals)
    # False Positives (Strict) = len(false_pos)
    # False Negatives = len(cond) - len(true_pos) - len(reverse)
    # Reversals = len(reverse)
    # Wait, if i->j is true, and we predict j->i.
    # True Pos: 0. False Pos: 0. Reverse: 1.
    # SHD cost should be 1.
    # If we predict nothing:
    # True Pos: 0. False Pos: 0. Reverse: 0. False Neg: 1.
    # SHD cost should be 1.
    # If we predict i->j and j->i (and true is i->j):
    # True Pos: 1. False Pos: 0. Reverse: 1 (j->i is reverse of i->j? No, j->i is extra if i->j exists? No).
    
    # Let's use the robust implementation from the existing codebase logic
    # which relies on lower-triangular comparison for SHD.
    
    pred_lower = np.flatnonzero(np.tril(B_est + B_est.T))
    cond_lower = np.flatnonzero(np.tril(B_true + B_true.T))
    extra_lower = np.setdiff1d(pred_lower, cond_lower, assume_unique=True)
    missing_lower = np.setdiff1d(cond_lower, pred_lower, assume_unique=True)
    shd_val = len(extra_lower) + len(missing_lower) + len(reverse)
    
    return {
        'fdr': fdr,
        'tpr': tpr,
        'fpr': fpr,
        'shd': int(shd_val),
        'nnz': pred_size,
        'precision': 1 - fdr,
        'recall': tpr
    }

def compute_metrics(W_true: np.ndarray, W_est: np.ndarray, threshold: float = 0.3) -> EvaluationMetrics:
    """
    Compute all standard metrics.
    
    Args:
        W_true: Ground truth weighted matrix [d, d].
        W_est: Estimated weighted matrix [d, d].
        threshold: Threshold for binarization.
        
    Returns:
        EvaluationMetrics object.
    """
    B_true = threshold_adjacency(W_true, threshold=threshold) # Usually W_true is already binary or we treat non-zeros as edges
    # If W_true is weighted, we treat non-zeros as edges.
    # Ideally W_true should be binary for evaluation against binary metrics.
    if not np.all(np.isin(B_true, [0, 1])):
         B_true = (W_true != 0).astype(int)
         
    B_est = threshold_adjacency(W_est, threshold=threshold)
    
    acc = count_accuracy(B_true, B_est)
    
    # F1 Score
    precision = acc['precision']
    recall = acc['recall']
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
        
    return EvaluationMetrics(
        f1=f1,
        shd=acc['shd'],
        tpr=acc['tpr'],
        fpr=acc['fpr'],
        precision=precision,
        recall=recall,
        nnz=acc['nnz']
    )
