# src/augmented_lagrangian.py

import torch
import torch.optim as optim
import time


def augmented_lagrangian_solver(
    X,
    h_func,
    lambda_l1=0.01,
    lr=1e-2,
    rho=1.0,
    max_iter=2000,
    device="cuda"
):
    """
    Unified augmented Lagrangian solver used for:
      - analytic polynomial constraint
      - neural surrogate constraint

    Always returns a dict:
        {
            "W_est": (d,d) matrix,
            "losses": list of floats,
            "runtime": seconds
        }
    """

    # Ensure X is on device
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    X = X.to(device)

    n, d = X.shape

    # Parameter to optimize
    W = torch.zeros((d, d), dtype=torch.float32, requires_grad=True, device=device)

    # Lagrange multiplier
    alpha = torch.tensor(0.0, device=device)

    optimizer = optim.Adam([W], lr=lr)

    losses = []
    h_history = []
    constraint_times = []
    t_start = time.perf_counter()

    for it in range(max_iter):

        optimizer.zero_grad()

        # Data-fitting loss
        F = (1/(2*n)) * torch.norm(X - X @ W, p='fro')**2
        F = F + lambda_l1 * torch.norm(W, 1)

        # Acyclicity constraint
        t0 = time.perf_counter()
        h = h_func(W)
        if torch.is_tensor(h):
             # Force synchronization for accurate timing if on GPU
            if h.device.type == 'cuda':
                torch.cuda.synchronize()
        t1 = time.perf_counter()
        constraint_times.append(t1 - t0)

        # Augmented Lagrangian
        L = F + alpha * h + 0.5 * rho * h**2

        L.backward()
        optimizer.step()

        # No self-loop
        with torch.no_grad():
            W.fill_diagonal_(0)

        # Update alpha every 200 iters
        if it % 200 == 0:
            alpha = alpha + rho * h.detach()

        losses.append(L.item())
        h_history.append(h.item())

    t_end = time.perf_counter()
    
    avg_constraint_time = sum(constraint_times) / len(constraint_times) if constraint_times else 0.0

    return {
        "W_est": W.detach().cpu(),
        "losses": losses,
        "h_history": h_history,
        "runtime": t_end - t_start,
        "avg_constraint_time": avg_constraint_time
    }
