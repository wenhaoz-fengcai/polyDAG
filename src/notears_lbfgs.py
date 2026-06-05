import numpy as np
import scipy.optimize as sopt
import torch

def notears_lbfgs(X, lambda1=0.1, loss_type='l2', max_iter=100, h_tol=1e-8, rho_max=1e+16, w_threshold=0.3):
    """
    Standard NOTEARS algorithm for linear SEM using PyTorch for GPU acceleration.
    Solves min_W L(W; X) + lambda1 ‖W‖_1 s.t. h(W) = 0.
    """
    # Detect device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Prepare data on device
    if isinstance(X, np.ndarray):
        X_torch = torch.from_numpy(X).float().to(device)
    elif isinstance(X, torch.Tensor):
        X_torch = X.float().to(device)
    else:
        X_torch = torch.tensor(X).float().to(device)

    n, d = X_torch.shape
    
    # Center the data
    X_torch = X_torch - torch.mean(X_torch, dim=0)
    
    def _h(W_torch):
        """Evaluate value and gradient of acyclicity constraint."""
        # h(W) = tr(e^{W*W}) - d
        M = W_torch * W_torch
        E = torch.matrix_exp(M)
        h = torch.trace(E) - d
        # nabla h(W) = (E.T * W * 2)
        G_h = E.t() * W_torch * 2
        return h, G_h

    def _loss(W_torch):
        """Evaluate value and gradient of loss."""
        M = X_torch @ W_torch
        if loss_type == 'l2':
            R = X_torch - M
            loss = 0.5 / n * torch.sum(R ** 2)
            G_loss = - 1.0 / n * X_torch.t() @ R
        elif loss_type == 'logistic':
            loss = 1.0 / n * torch.sum(torch.logaddexp(torch.zeros_like(M), M) - X_torch * M)
            G_loss = 1.0 / n * X_torch.t() @ (torch.sigmoid(M) - X_torch)
        elif loss_type == 'poisson':
            S = torch.exp(M)
            loss = 1.0 / n * torch.sum(S - X_torch * M)
            G_loss = 1.0 / n * X_torch.t() @ (S - X_torch)
        else:
            raise ValueError('unknown loss type')
        return loss, G_loss

    def _adj(w):
        """Convert doubled variables (w_pos, w_neg) back to W."""
        # w is numpy, convert to torch
        w_torch = torch.from_numpy(w).float().to(device)
        return (w_torch[:d*d] - w_torch[d*d:]).view(d, d)

    def _func(w):
        """Evaluate primal objective + augmented Lagrangian terms."""
        # w is numpy array from scipy optimizer
        W_torch = _adj(w)
        
        loss, G_loss = _loss(W_torch)
        h, G_h = _h(W_torch)
        
        # Objective = Loss + alpha*h + rho/2 * h^2 + lambda1 * |W|
        # L1 norm is handled by the bounds on w_pos, w_neg, so we add lambda1 * sum(w)
        w_torch = torch.from_numpy(w).float().to(device)
        obj = loss + 0.5 * rho * h * h + alpha * h + lambda1 * w_torch.sum()
        
        G_smooth = G_loss + (rho * h + alpha) * G_h
        
        # Gradient of L1 term is just lambda1 for all positive components
        g_obj_pos = G_smooth.view(-1) + lambda1
        g_obj_neg = -G_smooth.view(-1) + lambda1
        
        g_obj = torch.cat([g_obj_pos, g_obj_neg])
        
        return obj.item(), g_obj.cpu().numpy().astype('float64')

    # Bounds for w_pos and w_neg (>= 0)
    bnds = [(0, 0) if i == j else (0, None) for _ in range(2) for i in range(d) for j in range(d)]
    
    # Initialization
    w_est = np.zeros(2 * d * d)
    rho, alpha, h = 1.0, 0.0, np.inf
    
    for _ in range(max_iter):
        w_new, h_new = None, None
        while rho < rho_max:
            res = sopt.minimize(_func, w_est, method='L-BFGS-B', jac=True, bounds=bnds)
            w_new = res.x
            
            # Check constraint violation
            W_new_torch = _adj(w_new)
            h_new_val, _ = _h(W_new_torch)
            h_new = h_new_val.item()
            
            if h_new > 0.25 * h:
                rho *= 10
            else:
                break
        w_est, h = w_new, h_new
        alpha += rho * h
        if h <= h_tol or rho >= rho_max:
            break
            
    W_est = _adj(w_est).cpu().numpy()
    W_est[np.abs(W_est) < w_threshold] = 0
    return W_est
