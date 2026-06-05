# src/surrogate.py
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Callable, Optional


class SurrogateMLP(nn.Module):
    """
    Simple MLP surrogate:
    Input: flattened W (d*d)
    Output: scalar approximation of h_poly(W)
    """
    def __init__(self, d: int, hidden_dim: int = 512, num_layers: int = 3):
        super().__init__()
        in_dim = d * d
        layers = []
        dims = [in_dim] + [hidden_dim] * (num_layers - 1) + [1]

        for i in range(len(dims) - 2):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(dims[-2], dims[-1]))

        self.net = nn.Sequential(*layers)

    def forward(self, W: torch.Tensor) -> torch.Tensor:
        # W: [batch, d, d] or [d, d]
        if W.dim() == 2:
            W = W.unsqueeze(0)  # [1, d, d]
        B, d, _ = W.shape
        x = W.reshape(B, d * d)
        out = self.net(x)  # [B, 1]
        return out.squeeze(-1)  # [B]


def train_surrogate(
    d: int,
    h_poly_func: Callable[[torch.Tensor], torch.Tensor],
    num_samples: int = 5000,
    batch_size: int = 64,
    num_epochs: int = 50,
    lr: float = 1e-3,
    device: str = "cuda",
    save_path: Optional[str] = None,
):
    """
    Train a surrogate F_theta(W) ≈ h_poly(W).

    Args:
        d: number of nodes
        h_poly_func: function taking W[d,d] -> scalar h_poly(W)
        num_samples: total random W samples to generate
        batch_size: training batch size
        num_epochs: number of epochs
        lr: learning rate
        device: 'cuda' or 'cpu'
        save_path: where to save the trained surrogate (optional)
    """
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    model = SurrogateMLP(d=d).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    mse_loss = nn.MSELoss()

    # Pre-generate training data (simple random W; you可以之后换成随机 DAG)
    Ws = []
    ys = []
    with torch.no_grad():
        for _ in range(num_samples):
            # ---- 1) 生成较小分布的随机 W ----
            W = 0.1 * torch.randn(d, d)               # smaller scale
            W = W - torch.diag(torch.diag(W))         # zero diagonal

            # ---- 2) 控制谱半径，防止 (I-A) 接近奇异 ----
            A = (W * W).cpu()                         # A = W∘W
            eigenvals = torch.linalg.eigvals(A).abs()
            max_ev = eigenvals.max().item()

            if max_ev > 0.9:                          # 强行缩放谱半径
                W = W / (1.1 * max_ev)

            # ---- 3) 计算 h_poly / h_poly_geom ----
            h_val = h_poly_func(W.to(device)).detach().cpu()

            # ---- 4) 过滤非有限值（NaN, inf）----
            if not torch.isfinite(h_val):
                continue

            # ---- 5) 对 label 做限幅，避免极端梯度 ----
            h_val = torch.clamp(h_val, -100.0, 100.0)

            Ws.append(W)
            ys.append(h_val)


    Ws = torch.stack(Ws, dim=0)  # [N, d, d]
    ys = torch.stack(ys, dim=0)  # [N]
    N = Ws.size(0)

    print(f"[Surrogate] Training data: {N} samples, d={d}")

    for epoch in range(num_epochs):
        perm = torch.randperm(N)
        Ws = Ws[perm]
        ys = ys[perm]

        epoch_loss = 0.0
        num_batches = 0

        for idx in range(0, N, batch_size):
            W_batch = Ws[idx:idx + batch_size].to(device)
            y_batch = ys[idx:idx + batch_size].to(device)

            pred = model(W_batch)
            loss = mse_loss(pred, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        avg_loss = epoch_loss / max(1, num_batches)
        print(f"[Surrogate][Epoch {epoch+1}/{num_epochs}] loss = {avg_loss:.4e}")

    if save_path is not None:
        torch.save({"model_state": model.state_dict(), "d": d}, save_path)
        print(f"[Surrogate] Saved model to {save_path}")

    return model
