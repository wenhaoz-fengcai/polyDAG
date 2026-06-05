import torch
import torch.nn as nn
import torch.nn.functional as F


class GNNLayer(nn.Module):
    """
    Simple GraphSAGE-style GNN layer:
    h_i' = W1 * h_i + W2 * sum_{j ∈ N(i)} h_j
    """
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.W_self = nn.Linear(in_dim, out_dim, bias=False)
        self.W_neigh = nn.Linear(in_dim, out_dim, bias=False)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, H, A):
        """
        H: [d, F] or [B, d, F]
        A: [d, d] or [B, d, d] adjacency (weighted)
        """
        # Use Sum Aggregation (A @ H) for correct path counting
        if A.dim() == 3:
            neigh_agg = torch.bmm(A, H)
        else:
            neigh_agg = A @ H
        
        out = self.W_self(H) + self.W_neigh(neigh_agg)
        return self.norm(out)


class GNNSurrogate(nn.Module):
    """
    Permutation-equivariant surrogate for h(W)
    Input: W (d,d)
    Output: scalar surrogate h_hat(W)
    """
    def __init__(self, hidden_dim=64, num_layers=3, nonneg_output: bool = True):
        super().__init__()
        self.input_dim = 1  # edge weight scalar
        self.nonneg_output = nonneg_output

        layers = []
        for _ in range(num_layers):
            layers.append(GNNLayer(self.input_dim if _ == 0 else hidden_dim,
                                   hidden_dim))
        self.layers = nn.ModuleList(layers)

        # graph-level readout
        self.readout_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, W):
        """
        W: [d, d] or [B, d, d]
        """
        if W.dim() == 2:
            # Node feature = row-sum or column-sum? → use both (2 features)
            # FIX: Use absolute sum to avoid cancellation (e.g. 0.5 + -0.5 = 0)
            degree = W.abs().sum(dim=1, keepdim=True)  # [d,1]
            H = degree  # initial node embedding

            # Use weighted adjacency to preserve edge strength
            A = W.abs()

            for layer in self.layers:
                H = layer(H, A)
                H = F.relu(H)

            # graph-level pooling
            g = H.mean(dim=0)              # [F]

            # final scalar output
            out = self.readout_mlp(g).squeeze()
        elif W.dim() == 3:
            degree = W.abs().sum(dim=2, keepdim=True)  # [B,d,1]
            H = degree
            A = W.abs()

            for layer in self.layers:
                H = layer(H, A)
                H = F.relu(H)

            g = H.mean(dim=1)  # [B,F]
            out = self.readout_mlp(g).squeeze(-1)  # [B]
        else:
            raise ValueError(f"Expected W to have dim 2 or 3, got {W.dim()}")

        if self.nonneg_output:
            # Backward-compatible behavior: enforce non-negativity
            return F.softplus(out)
        return out
