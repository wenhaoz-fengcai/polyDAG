"""
benchmark_poly_variants.py
--------------------------
Comprehensive benchmark comparing three acyclicity constraint variants:
  - notears_exp    : h_exp(W) = tr(exp(W⊙W)) - d   [NOTEARS baseline]
  - poly_direct    : h_poly(W) = sum_{k=1}^d tr((W⊙W)^k)  [direct loop]
  - poly_geometric : h_poly via geometric-series solve  [efficient variant]

Evaluated across graph sizes d = [10, 20, 50, 100] with n_seeds=5 per size.

Outputs
-------
results/poly_variants_TIMESTAMP/
    all_results.json   — flat list of all run dicts
    seed_X/d_Y/        — per-run W_est.npy, W_true.npy, h_history.npy
"""

import sys
import os
import json
import time
import datetime
import numpy as np
import torch

# Add project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.constraints import h_poly, h_poly_noloop_geometric, h_exp
from src.augmented_lagrangian import augmented_lagrangian_solver
from src.sem_generator import generate_er_graph, assign_edge_weights, sample_linear_sem
from src.metrics import compute_metrics, threshold_adjacency

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────

D_VALUES   = [10, 20, 50, 100]
N_SAMPLES  = 1000
N_SEEDS    = 5
LAMBDA_L1  = 0.01
MAX_ITER   = 2000
ER_P       = 4  # expected degree (edges per node) — used as k in ER(d, k/d)

METHODS = {
    "notears_exp": lambda W: h_exp(W),
    "poly_direct": lambda W: h_poly(W),
    "poly_geometric": lambda W: h_poly_noloop_geometric(W),
}


def er_edge_prob(d: int, expected_degree: int = 4) -> float:
    """Return the ER edge probability for expected in-degree `expected_degree`."""
    return min(expected_degree / max(d - 1, 1), 1.0)


def run_one(d: int, seed: int, method_name: str, h_func, device: str, out_dir: str):
    """Run one (d, seed, method) trial and return a result dict."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Data generation
    p = er_edge_prob(d, ER_P)
    B_true = generate_er_graph(d, p=p)
    W_true = assign_edge_weights(B_true)
    X = sample_linear_sem(W_true, n=N_SAMPLES)

    # Solve
    res = augmented_lagrangian_solver(
        X,
        h_func=h_func,
        lambda_l1=LAMBDA_L1,
        max_iter=MAX_ITER,
        device=device,
    )

    W_est = res["W_est"].numpy()
    W_true_np = W_true.numpy()

    # Save arrays
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "W_est.npy"), W_est)
    np.save(os.path.join(out_dir, "W_true.npy"), W_true_np)
    np.save(os.path.join(out_dir, "h_history.npy"), np.array(res["h_history"]))

    # Metrics
    W_true_bin = (np.abs(W_true_np) > 0).astype(int)
    metrics = compute_metrics(W_true_bin, W_est)

    return {
        "d": d,
        "n": N_SAMPLES,
        "p": p,
        "seed": seed,
        "method": method_name,
        "f1": metrics.f1,
        "shd": metrics.shd,
        "tpr": metrics.tpr,
        "fpr": metrics.fpr,
        "precision": metrics.precision,
        "recall": metrics.recall,
        "runtime": res["runtime"],
        "avg_constraint_time": res["avg_constraint_time"],
        "h_final": float(res["h_history"][-1]),
    }


def main():
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base_out = os.path.join(
        os.path.dirname(__file__), "..", "results", f"poly_variants_{timestamp}"
    )
    os.makedirs(base_out, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Output: {base_out}\n")

    all_results = []

    for d in D_VALUES:
        for seed in range(1, N_SEEDS + 1):
            for method_name, h_func in METHODS.items():
                out_dir = os.path.join(
                    base_out, f"d{d}", f"seed{seed}", method_name
                )
                print(f"  d={d:4d}  seed={seed}  method={method_name:<18}", end="  ", flush=True)
                t0 = time.perf_counter()
                try:
                    rec = run_one(d, seed, method_name, h_func, device, out_dir)
                    all_results.append(rec)
                    print(
                        f"F1={rec['f1']:.3f}  SHD={rec['shd']:4d}  "
                        f"time={rec['runtime']:.2f}s  h_final={rec['h_final']:.2e}"
                    )
                except Exception as exc:
                    print(f"ERROR: {exc}")
                    all_results.append({
                        "d": d, "seed": seed, "method": method_name, "error": str(exc)
                    })

    out_json = os.path.join(base_out, "all_results.json")
    with open(out_json, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nSaved {len(all_results)} records → {out_json}")


if __name__ == "__main__":
    main()
