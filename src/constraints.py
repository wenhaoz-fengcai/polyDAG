import torch


def h_poly(W, K=None):
    """
    Polynomial acyclicity constraint:
      h(W) = sum_{k=1}^K tr((W⊙W)^k)
    Default K = d.
    """
    A = W * W
    d = A.size(0)
    if K is None:
        K = d

    B = A.clone()
    h = torch.trace(B)

    for _ in range(2, K + 1):
        B = B @ A
        h = h + torch.trace(B)

    return h

def h_poly_noloop_geometric(W, K=None):
    r"""
    Polynomial acyclicity constraint using a matrix geometric series:

    .. math::
        \sum_{k=1}^{K} A^{k}
        = (I - A)^{-1}\left(A - A^{K+1}\right),

    where :math:`A = W \circ W`.
    """
    A = W * W
    d = A.size(0)
    if K is None:
        K = d

    I = torch.eye(d, device=W.device, dtype=W.dtype)

    AK1 = torch.matrix_power(A, K + 1)
    S = torch.linalg.solve(I - A, A - AK1)

    return torch.trace(S)

def h_exp(W):
    """
    NOTEARS exponential constraint:
        h(W) = tr(exp(W⊙W)) - d
    """
    A = W * W
    d = A.size(0)
    return torch.trace(torch.linalg.matrix_exp(A)) - d
