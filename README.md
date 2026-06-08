# polyDAG: Polynomial Acyclicity Constraints for Efficient Continuous Causal Discovery in Visual Semantic Graphs

This repository contains the code and resources for the paper "**polyDAG: Polynomial Acyclicity Constraints for Efficient Continuous Causal Discovery in Visual Semantic Graphs**."

> **TL;DR**: We introduce a novel, scalable, and trace-based polynomial acyclicity constraint for continuous DAG learning. This README explains the study context, provides the source code, and shows how to reproduce our synthetic benchmarks.

## Background & Objectives

**Background**: Evaluating acyclicity in differentiable directed acyclic graph (DAG) learning typically relies on the trace of a matrix exponential, which demands computationally expensive $O(d^3)$ approximations (e.g., scaling-and-squaring Padé) at every gradient step.

**Objective**: Develop a finite polynomial acyclicity constraint that requires fewer operations while remaining theoretically exact. We formulate a geometric-series evaluation (`polyDAG-Geo`) that reduces the sequential loops to a single matrix linear solve and matrix power, speeding up structure learning for visual semantic graphs and broader causal discovery tasks.

## Key Results (from the paper)

1. **Faster end-to-end structure learning**: At 100 nodes, the geometric variant (`polyDAG-Geo`) runs in 3.44 seconds compared with 5.16 seconds for the exponential baseline (NOTEARS-Exp), achieving a **~33% speedup**. The speed up trends remain steady across $d \in \{100, 200, 500\}$.
2. **Improved Structure Recovery**: Over the synthetic protocol ($d \in \{100,200,500\}$), polyDAG reduces the mean Structural Hamming Distance (SHD) from 318.4 to 285.4 and improves mean F1 score from 0.725 to 0.756 compared to the baseline.

**Conclusion**: The geometric polynomial constraint serves as a faster, identically exact substitute for the matrix-exponential constraint, empowering more scalable visual causal learning and data analysis pipelines.

## Why this matters

- **Computational Efficiency**: Avoids slow dense scaling-and-squaring overhead.
- **Identical Feasible Set**: Provably defines the exact same DAG space as the NOTEARS exponential constraint.
- **Easy Integration**: Acts as a drop-in regularization constraint in existing augmented Lagrangian continuous causal discovery frameworks.

## Repository Layout

- `src/` - The core library including implementations for the polynomial constraint (`constraints.py`), augmented Lagrangian optimizer, and data generation/metrics.
- `benchmark_synthetic.py` - The main driver script to reproduce the synthetic benchmark results (Table 4, Figure 2) measuring SHD, F1, and runtime.

### Note

> **The CelebA visual attribute experiment data has been excluded from this minimal reproduction code due to storage constraints. This package provides the solver logic and synthetic data engines sufficient to verify the theorems and latency scaling results (Table 4 & Figure 2).**

## Installation

We recommend Python 3.9 or above with the following basic dependencies:

```bash
pip install torch numpy scipy networkx
```

## Running the benchmarks

To reproduce the efficiency and structure recovery results (Table 4 and runtime data for Figure 2):

```bash
python benchmark_synthetic.py
```

Arguments (e.g. graph size $d$, seeds, hyperparameters) can be configured within the script to execute specific runs across ER-sparse, ER-dense, or scale-free matrices.

## Citing

If you use this code or model in your research, please cite our paper:

```bibtex
@article{zhang2026polydag,
    author = {Zhang, Wenhao and Ramezani, Ramin and Han, Tao and Hwang, Kai and Guo, Minyi},
    title = {polyDAG: Polynomial Acyclicity Constraints for Efficient Continuous Causal Discovery in Visual Semantic Graphs},
    year = {2026},
    journal = {The Visual Computer},
    archivePrefix = {arXiv},
    arxivId = {2606.06908},
    url = {https://arxiv.org/pdf/2606.06908}
}
```

## License

Code: see LICENSE (MIT) in this repository.

## Contact 

For questions, please open an issue or contact Dr. Wenhao Zhang (zhang.wenhao@sjtu.edu.cn).
