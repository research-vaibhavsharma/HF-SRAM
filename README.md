# HF-SRAM
Official implementation of "Linear Convergence of Hessian-Free Federated Meta-Learning via Swarm-Regularized ADMM" (HF-SRAM). Features experiments on CIFAR-100, FEMNIST, and non-linear regression under severe non-IID regimes.

# HF-SRAM: Hessian-Free Swarm-Regularized ADMM for Federated Meta-Learning

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Framework: PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)

This repository contains the official source code, experiment configurations, and supplementary mathematical proofs for the paper: **"Linear Convergence of Hessian-Free Federated Meta-Learning via Swarm-Regularized ADMM."**

> **⚠️ UNDER REVIEW NOTICE**
>
> This manuscript is currently under peer review. To comply with journal policies and accommodate ongoing revisions, the complete execution pipeline (including the full `main.py` orchestrator and baseline integrations) will be officially released upon the paper's acceptance. 
> 
> *Please note: The current codebase is provided for structural and mathematical reference. It is subject to active modifications, optimizations, and updates during the review process.*

## 📖 Overview

Federated Meta-Learning (FML) enables rapid model adaptation across distributed networks but is traditionally bottlenecked by the intractable computation of second-order Hessian matrices. While first-order approximations bypass these costs, they inject persistent structural biases and suffer catastrophic client drift under severe non-IID regimes.

**HF-SRAM** resolves these bottlenecks by structurally embedding the meta-objective into an Alternating Direction Method of Multipliers (ADMM) framework and anchoring primal updates with evolutionary swarm intelligence. 

### Key Contributions
* **Hessian-Free Efficiency:** Achieves edge-friendly $\mathcal{O}(d)$ computational complexity without relying on complex Neumann series or Gaussian Processes.
* **Bounded Variance Relaxation:** Structurally absorbs data heterogeneity via a dual multiplier, guaranteeing linear convergence under the Polyak-Łojasiewicz (PŁ) condition *without* the restrictive bounded stochastic variance assumption.
* **State-of-the-Art Performance:** Outperforms cutting-edge 2024–2026 baselines (FedConv, FedSAM, FedAKT) by over 2.85% absolute accuracy under severe Dirichlet non-IID partitioning ($\alpha_{\text{Dir}}=0.1$).

---

## 🚀 Implemented Baselines

The repository includes implementations for our proposed method and six state-of-the-art federated optimization frameworks:
1. **HF-SRAM** (Ours)
2. **FedAKT** (2026) - Federated Adaptive Knowledge Transfer
3. **FedBNR** (2026) - Federated Bayesian Neural Regression
4. **FedPAC** (2025) - PAC-Bayesian Federated Learning
5. **FedSAM** (2024) - Federated Sharpness-Aware Minimization
6. **FedConv** (2024) - Federated Convergence Acceleration
7. **FedDE** (2024) - Federated Dual-Ensemble

@article{sharma2026hfsram,
  title={Linear Convergence of Hessian-Free Federated Meta-Learning via Swarm-Regularized ADMM},
  author={Sharma, Vaibhav and Pardasani, Kamal Raj},
  journal={Submitted to IEEE Signal Processing Letters},
  year={2026}
}
