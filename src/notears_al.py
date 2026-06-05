# src/notears_al.py

import torch
from src.constraints import h_exp
from src.augmented_lagrangian import augmented_lagrangian_solver


def notears_al(
    X,
    lambda_l1=0.01,
    device="cuda",
):
    """
    NOTEARS exponential constraint baseline:
        min_W F(W) s.t. h_exp(W) = 0
    Uses the same augmented Lagrangian solver as polynomial + surrogate versions.
    """
    return augmented_lagrangian_solver(
        X,
        h_func=lambda W: h_exp(W),
        lambda_l1=lambda_l1,
        device=device,
    )
