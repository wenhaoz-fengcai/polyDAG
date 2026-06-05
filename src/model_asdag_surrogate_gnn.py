# src/model_asdag_surrogate_gnn.py

import time
from typing import Optional, Dict, Any

import torch
import torch.optim as optim
import numpy as np
import torch.nn.functional as nnF

from src.surrogate_gnn import GNNSurrogate


def learn_asdag_surrogate_gnn(
    X: torch.Tensor,
    lambda_l1: float = 0.01,
    device: str = "cuda",
    max_iter: int = 100, # Inner iterations per ALM step
    rho: float = 1.0,
    alpha_init: float = 0.0,
    surrogate: Optional[GNNSurrogate] = None,
    surrogate_ckpt: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Causal discovery with augmented Lagrangian +
    GNN surrogate F_theta(W) ≈ h_poly(W).

    Args:
        X: data matrix [n, d]
        lambda_l1: L1 penalty
        device: 'cuda' or 'cpu'
        max_iter: optimization steps per ALM iteration
        rho: initial penalty parameter
        alpha_init: initial Lagrange multiplier
        surrogate: pre-initialized GNN surrogate (optional)
        surrogate_ckpt: checkpoint path (optional)

    Returns:
        dict with W_est, runtime, loss history
    """

    device = torch.device(device if torch.cuda.is_available() else "cpu")
    X = X.to(device)
    n, d = X.shape

    # ---------------------------------------------------------
    # Load or create surrogate
    # ---------------------------------------------------------
    if surrogate is None:
        if surrogate_ckpt is None:
            raise ValueError("Either surrogate or surrogate_ckpt must be provided.")
        ckpt = torch.load(surrogate_ckpt, map_location=device)

        hidden_dim = ckpt.get("hidden_dim", 64)   # fallback
        num_layers = ckpt.get("num_layers", 3)
        nonneg_output = ckpt.get("nonneg_output", True)

        surrogate = GNNSurrogate(hidden_dim=hidden_dim, num_layers=num_layers, nonneg_output=nonneg_output)
        surrogate.load_state_dict(ckpt["model_state"])
        surrogate.to(device)

        # Check for new normalization params
        mean_y = ckpt.get("mean_y", 0.0)
        std_y = ckpt.get("std_y", 1.0)
        log_transform = ckpt.get("log_transform", False)
        scale = ckpt.get("scale", 1.0) # Legacy support
    else:
        # if caller provides surrogate directly
        scale = 1.0
        mean_y = 0.0
        std_y = 1.0
        log_transform = False

    surrogate.eval()

    # Keep W inside the surrogate's training regime to avoid extreme extrapolation
    # that can overflow the inverse log transform.
    w_clamp = 0.5

    # ---------------------------------------------------------
    # Parameter W to optimize
    # ---------------------------------------------------------
    # Initialize with larger random noise to ensure gradients flow through GNN
    # The surrogate was trained on W ~ 0.2 * randn, so we need comparable magnitude
    W = torch.zeros(d, d, device=device, requires_grad=True)
    with torch.no_grad():
        W.uniform_(-0.1, 0.1)
        W.fill_diagonal_(0)

    optimizer = optim.Adam([W], lr=1e-2) # Adam is robust
    
    alpha = alpha_init
    rho_max = 1e10 # Cap rho to prevent instability
    h_tol = 1e-8
    
    losses = []
    h_history = []
    constraint_times = []
    t_start = time.perf_counter()

    # ---------------------------------------------------------
    # Data fitting objective + L1
    # ---------------------------------------------------------
    def F(W_mat: torch.Tensor) -> torch.Tensor:
        XW = X @ W_mat
        mse = 0.5 / n * torch.sum((X - XW) ** 2)
        l1 = lambda_l1 * torch.sum(torch.abs(W_mat))
        return mse + l1

    # ---------------------------------------------------------
    # Augmented Lagrangian optimization Loop
    # ---------------------------------------------------------
    # We use a simplified ALM loop:
    # Outer loop: update rho, alpha
    # Inner loop: optimize W
    
    h_val = np.inf
    
    for outer_it in range(20): # Max 20 ALM steps
        
        h_new = np.inf
        
        # Inner optimization loop
        for inner_it in range(max_iter):
            optimizer.zero_grad()

            loss_data = F(W)

            # surrogate acyclicity
            t0 = time.perf_counter()
            h_sur_raw = surrogate(W) 
            
            if log_transform:
                # Inverse of z = log1p(h):
                #   log_h = z = h_sur_raw*std + mean
                # Ensure non-negativity and numerical stability.
                log_h = h_sur_raw * std_y + mean_y
                # IMPORTANT: do not use softplus here.
                # softplus(0)=0.693 -> expm1(0.693)≈1.0, which prevents h from reaching 0.
                log_h = torch.clamp(log_h, min=0.0)
                log_h = torch.clamp(log_h, max=20.0)
                h_sur = torch.expm1(log_h)
            else:
                h_sur = scale * h_sur_raw

            # Ensure non-negative (for non-log checkpoints)
            if not log_transform:
                h_sur = nnF.softplus(h_sur)
            
            if h_sur.device.type == 'cuda':
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            constraint_times.append(t1 - t0)

            # augmented Lagrangian
            loss_aug = loss_data + alpha * h_sur + 0.5 * rho * (h_sur ** 2)

            loss_aug.backward()
            optimizer.step()

            # No self-loops
            with torch.no_grad():
                W.data -= torch.diag(torch.diag(W.data))
                W.data.clamp_(-w_clamp, w_clamp)

            losses.append(loss_aug.item())
            h_history.append(h_sur.item())
            h_new = h_sur.item()

        print(
            f"[GNN-SurrogateCD][Outer {outer_it}] "
            f"F={loss_data.item():.4f}, "
            f"h_sur={h_new:.4e}, "
            f"rho={rho:.1e}, "
            f"alpha={alpha:.1e}"
        )
        
        # Update rho and alpha
        if h_new > 0.25 * h_val:
            rho *= 5 # Gentler update
        else:
            # Only update alpha if we made progress on h
            pass
            
        h_val = h_new
        alpha += rho * h_val
        
        if h_val <= h_tol or rho >= rho_max:
            break

    t_end = time.perf_counter()
    
    avg_constraint_time = sum(constraint_times) / len(constraint_times) if constraint_times else 0.0

    return {
        "W_est": W.detach().cpu(),
        "losses": losses,
        "h_history": h_history,
        "runtime": t_end - t_start,
        "avg_constraint_time": avg_constraint_time
    }
