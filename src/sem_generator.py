import torch
import numpy as np


def generate_er_graph(d, p=0.1):
    """
    Generate an Erdos–Renyi DAG by:
    1) sampling an ER adjacency matrix
    2) applying a random permutation
    3) taking strictly upper triangular part
    """
    A = (np.random.rand(d, d) < p).astype(float)
    np.fill_diagonal(A, 0)

    # random topological order
    P = np.random.permutation(d)
    A = A[P][:, P]

    # DAG: keep upper triangular only
    A = np.triu(A, k=1)
    return torch.tensor(A, dtype=torch.float32)


def assign_edge_weights(A, low=0.5, high=2.0):
    """
    Assign random weights to edges in A.
    """
    d = A.size(0)
    W = torch.zeros_like(A)
    weights = (torch.rand(d, d) * (high - low) + low) * torch.sign(torch.rand(d, d) - 0.5)
    W = A * weights
    return W


def sample_linear_sem(W, n=1000, noise_scale=1.0):
    """
    Sample X from X = XW + Z  → X = Z (I - W)^{-1}
    """
    d = W.size(0)
    Z = noise_scale * torch.randn(n, d)

    I = torch.eye(d, dtype=torch.float32)
    X = Z @ torch.linalg.inv(I - W)

    return X


def sample_nonlinear_sem_gcastle(W, n=1000, sem_type='mlp'):
    """
    Generate non-linear data using gcastle.
    W: Adjacency matrix (weighted or binary).
    sem_type: 'mlp', 'mim', 'gp', 'gp-add'
    """
    try:
        from castle.datasets import IIDSimulation
        from castle.common import GraphDAG
    except ImportError:
        print("gcastle not installed. Please install it via `pip install gcastle`.")
        return None

    if isinstance(W, torch.Tensor):
        W = W.detach().cpu().numpy()
    
    # gcastle expects binary adjacency for structure, 
    # but IIDSimulation usually generates its own weights or takes W.
    # IIDSimulation(W=W, n=n, method=sem_type, sem_type=sem_type)
    
    # Note: IIDSimulation in gcastle usually takes W as the ground truth graph.
    # If W is weighted, it might treat it as weighted or binary depending on implementation.
    # Let's assume it takes the structure.
    
    # IIDSimulation signature:
    # __init__(self, W, n=1000, method='linear', sem_type='gauss', noise_scale=1.0)
    # method: 'linear', 'nonlinear'
    # sem_type: 'mlp', 'mim', 'gp', 'gp-add' (if method='nonlinear')
    
    sim = IIDSimulation(W=W, n=n, method='nonlinear', sem_type=sem_type)
    X = sim.X
    
    return torch.tensor(X, dtype=torch.float32)

