
#  A GPU solver combined rAPDHG with LBFGS-B for Nonlinear Programming with Linear Constraints

[](https://www.google.com/search?q=) [](https://www.google.com/search?q=/LICENSE)

This repository contains a high-performance, GPU-accelerated solver for large-scale **Nonlinear Programming (NLP)** problems with linear equality and box constraints. It leverages a **Restarted Anderson-accelerated Primal-Dual Hybrid Gradient (rAPDHG)** scheme, using **L-BFGS-B** to efficiently solve the primal subproblem.

-----

## 🚧 Current Status

**This project is under active development.** The current version is a proof-of-concept tailored specifically for the **Fisher market equilibrium problem with CES utilities**. A general-purpose API is planned for a future release.

If you have questions or suggestions, please open an issue or email me at `ishongpeili@gmail.com`.

-----

## ✨ Key Features

  * **GPU Accelerated:** Designed from the ground up for massive parallelism on GPUs to handle large-scale problems.
  * **Efficient Subproblem Solver:** Integrates L-BFGS-B to exactly and efficiently solve the primal subproblem, including box constraints. This improves robustness compared to methods that only approximate the solution.
  * **Advanced PDHG Scheme:** Employs a Restarted Anderson-accelerated Primal-Dual Hybrid Gradient method for faster convergence.

-----

## Problem Formulation

The solver is designed for problems of the form:

$$
\begin{aligned}
& \min_{x} & & f(x) \\
& \text{s.t.} & & Ax = b \\
& & & l \leq x \leq u
\end{aligned}
$$where $f(x)$ is a smooth, convex function.

The core of our method is the Primal-Dual Hybrid Gradient (PDHG) algorithm. The update scheme involves solving a primal subproblem for $x$ and then performing a dual gradient ascent step for $y$:

$$

\begin{aligned}

x^{k+1} &= \argmin_{l\leq x \leq u} f(x) + y^{\top}(Ax-b) + \frac{1}{2\tau^{k+1}}||x-x^{k-1}||_2^2\\

y^{k+1} &= y^{k} + \delta^{k+1}(A(2x^{k+1} - x^k) - b)

\end{aligned}

$$

- The **primal subproblem ($x^{k+1}$)** is a box-constrained nonlinear optimization problem, which we solve exactly using a GPU-accelerated L-BFGS-B algorithm.

- The **dual update ($y^{k+1}$)** includes an extrapolation step $\theta\_k$ for acceleration.

-----

## Implementation

We adapted the excellent [**cuLBFGSB**](https://github.com/raymondyfei/lbfgsb-gpu) library, a CUDA implementation of L-BFGS-B, to serve as our primal subproblem solver. This allows us to handle box constraints directly within the subproblem, which is numerically superior to treating them as dualized constraints.

-----

## 🚀 Performance: Fisher Market Equilibrium

We benchmarked our solver against [**Mosek**](https://www.mosek.com/), a state-of-the-art commercial conic optimizer, on the Fisher market equilibrium problem with CES utilities.

### Formulation
The problem is formulated as:

$$
\begin{aligned}
& \min_{x} && -\sum_{i=1}^{n} w_i \log \left( \sum_{j=1}^{m} u_{ij} x_{ij}^p \right)^{1/p} \\
& \text{s.t.} && \sum_{i=1}^{n} x_{ij} = 1, \quad \forall j \in [m] \\
&&& x_{ij} \geq 0, \quad \forall i,j
\end{aligned}
$$

### Conic Formulation 
To solve this problem by mosek, we first reformulate it to a conic programming:
$$



\begin{aligned}

\min_{x,t,y,s}\quad & -\sum_{i=1}^{n} w_i s_i \\[4pt]

\text{s.t.}\quad

& \sum_{i=1}^{n} x_{ij} = 1 && \forall j\in[m] \\[4pt]

& \sum_{j=1}^{m} u_{ij}\,t_{ij} = y_{i} && \forall i\in[n] \\[4pt]

& x_{ij}\ge 0 && \forall i,j \\[4pt]

& (x_{ij},1,t_{ij})\in\mathcal{P}^{p,1-p}_{\text{pow},3} && \forall i,j \\[4pt]

& (y_{i},1,s_i)\in\mathcal{P}_{\text{exp},3} && \forall i

\end{aligned}

$$

### Stop Criterion
The relative residual $r$ is defined as $r= \max(r_{\text{primal}}, r_{\text{dual}})$, and $r_{\text{primal}}$ and $r_{\text{dual}}$ are defined as:
$$
r_{\text{primal}}
= \frac{\left\| \mathbf{1}_{n}^{T} x - 1 \right\|_{\infty}}
       {1 + \max \left\{ \left\| \mathbf{1}_{n}^{T} x \right\|_{\infty}, 1 \right\}}

$$
$$

r_{\text{dual}}
= \frac{\left\| g + p_j^{k+1} \right\|_{\infty}}
       {1 + \max \left\{ \left\| g \right\|_{\infty},\, p_j^{k+1} \right\}},
\quad
g = -\frac{ w_i u_{ij} x_{ij}^{\,p-1} }
         { \sum_{j\in[m]} u_{ij} x_{ij}^p }
$$

By setting $p=0.5$ and $r<1e-4$, we conduct a experiment over different scales of problems compared with Mosek.
The results below demonstrate a significant performance advantage for our solver as the problem size scales.


| Agents (n) | Goods (m) | Variables (n\*m) | Mosek Time (s) | **This Project (s)** | Speedup |
| :--------- | :-------- | :-------------- | :------------- | :------------------- | :------ |
| 1,000      | 400       | 80,000          | 1.60           | **1.33** | 1.2x    |
| 10,000     | 4,000     | 800,000         | 29.13          | **2.75** | 10.6x   |
| 100,000    | 4,000     | 8,000,000       | 218.56         | **9.48** | 23.1x   |
| 100,000    | 5,000     | 10,000,000      | 302.05         | **20.98** | 14.4x   |
| 1,000,000  | 10,000    | 20,000,000      | 556.57         | **87.61** | 6.4x    |

-----

## 🗺️ Roadmap

- [x] Solver for Fisher Problem with CES utility.
- [ ] Develop a general-purpose API for convenient usage.
- [ ] Add support for more problem types and constraints.
- [ ] Publish comprehensive documentation.
$$