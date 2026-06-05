# src/model_polycd.py

import torch
from src.constraints import h_poly, h_poly_noloop_geometric
from src.augmented_lagrangian import augmented_lagrangian_solver


def learn_polycd(X, lambda_l1=0.01, device="cuda", K=None):
    """
    Main function to learn a DAG using the polynomial constraint.
    """
    return augmented_lagrangian_solver(
        X,
        lambda W: h_poly_noloop_geometric(W, K=K),
        lambda_l1=lambda_l1,
        device=device
    )
