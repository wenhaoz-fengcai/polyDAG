# src/model_asdag_surrogate_mlp.py
import time
from typing import Dict, Any, Optional

import torch
import torch.nn as nn
import torch.optim as optim

from .surrogate import SurrogateMLP


def learn_asdag_surrogate_mlp(
    X: torch.Tensor,
    lambda_l1: float = 0.01,
    device: str = "cuda",
    max_iter: int = 200,
    rho: float = 1.0,
    alpha_init: float = 0.0,
    surrogate: Optional[SurrogateMLP] = None,
    surrogate_ckpt: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Causal discovery with augmented Lagrangian +
    neural surrogate F_theta(W) ≈ h_poly(W).

    Args:
        X: data matrix [n, d]
        lambda_l1: l1 penalty
        device: 'cuda' or 'cpu'
        max_iter: optimization steps
        rho: penalty parameter
        alpha_init: initial Lagrange multiplier
        surrogate: pre-initialized SurrogateMLP (optional)
        surrogate_ckpt: path to a saved surrogate checkpoint (optional)

    Returns:
        dict with W_est, runtime, loss history, etc.
    """
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    X = X.to(device)
    n, d = X.shape

    # Load or create surrogate
    if surrogate is None:
        if surrogate_ckpt is None:
            raise ValueError("Either surrogate or surrogate_ckpt must be provided.")
        ckpt = torch.load(surrogate_ckpt, map_location=device)
        surrogate = SurrogateMLP(d=ckpt["d"])
        surrogate.load_state_dict(ckpt["model_state"])
        surrogate.to(device)
    surrogate.eval()

    # Parameter W to optimize
    W = torch.zeros(d, d, device=device, requires_grad=True)

    optimizer = optim.Adam([W], lr=1e-2)
    alpha = alpha_init

    # Data-fitting loss + L1
    def F(W_mat: torch.Tensor) -> torch.Tensor:
        # squared loss + l1
        XW = X @ W_mat
        mse = 0.5 / n * torch.sum((X - XW) ** 2)
        l1 = lambda_l1 * torch.sum(torch.abs(W_mat))
        return mse + l1

    losses = []
    t_start = time.perf_counter()

    for it in range(max_iter):
        optimizer.zero_grad()

        loss_data = F(W)
        # surrogate acyclicity
        h_sur = surrogate(W.unsqueeze(0))  # [1] -> scalar
        # augmented Lagrangian (based on surrogate)
        loss_aug = loss_data + alpha * h_sur + 0.5 * rho * h_sur ** 2

        loss_aug.backward()
        optimizer.step()

        # zero diagonal constraint (no self-loops)
        with torch.no_grad():
            W.data -= torch.diag(torch.diag(W.data))

        losses.append(loss_aug.item())

        if it % 50 == 0 or it == max_iter - 1:
            print(f"[Surrogate-CD][Iter {it}/{max_iter}] "
                  f"F={loss_data.item():.4f}, h_sur={h_sur.item():.4e}, "
                  f"loss={loss_aug.item():.4f}")

        # Optionally update alpha or rho (简单版先不动)
        # alpha = alpha + rho * h_sur.item()

    t_end = time.perf_counter()

    return {
        "W_est": W.detach().cpu(),
        "losses": losses,
        "runtime": t_end - t_start,
    }
