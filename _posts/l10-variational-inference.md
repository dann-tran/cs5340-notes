---
title: Variational Inference
tag: approximate inference
lectureNumber: 10
---

## Inference as Optimization

The main idea of variational methods is to cast inference as an optimization problem.

Suppose we are given an intractable probability distribution $p$. Variational techniques will try to solve an optimization problem over a class of tractable distributions $\mathcal{Q}$ in order to find a $q\in\mathcal{Q}$ that is most similar to $p$. We will then query $q$ (rather than $p$) in order to get an approximate solution.

Main differences between sampling and variational techniques:

- Unlike sampling-based methods, variational approaches will almost never find the globally optimal solution.
- However, we will always know if they have converged. In some cases, we will even have bounds on their accuracy.
- In practice, variational inference methods often scale better and are more amenable to techniques like stochastic gradient optimization, parallelization over multiple processors, and acceleration using GPUs.

Setup:

- Observed data $\mathcal D$.
- Latent, unobserved variables $\mathbf x$.
- Target distribution is the posterior $p(\mathbf x)=p(\mathbf x|\mathcal D)$.
- Unnormalized target distribution is the joint density $\tilde p(\mathbf x)=p(\mathbf x, \mathcal D)$.
- Normalisation constant is the likelihood $Z=p(\mathcal D)=\int_{\mathcal Z} p(\mathbf x, \mathcal D)$

### The Kullback-Leibler (KL) divergence

To formulate inference as an optimization problem, we need to choose an approximating family $Q$ and an optimization objective $J(q)$, for which the KL divergence is a candidate. The KL divergence between two distributions $q$ and $p$ with discrete support is defined as

$$
\text{KL}(q\|p)=\sum_{\mathbf x} q(x)\log\frac{q(\mathbf x)}{p(\mathbf x)}=\mathbb{E}_q\left[\log\frac{q(\mathbf x)}{p(\mathbf x)}\right]
$$

Properties:

- $KL(q\|p)\geq 0 \space \forall q, p$.
- $KL(q\|p)=0\iff q=p$
- $KL(q\|p)\neq KL(p\|q)$ (i.e. the KL divergence is not symmetric, hence not a distance)

### The variational lower bound

Assume that $p$ can be evaluated up to a normalization constant. Optimizing $KL(q\|p)$ directly is not possible because of the potentially intractable normalization constant $Z$. Instead, we'll work with the unnormalized probability $\tilde p$

$$
J(q)\triangleq\text{KL}(q\|\tilde{p})=\sum_{\mathcal Z} q(\mathbf x)\log\frac{q(\mathbf x)}{\tilde{p}(\mathbf x)}
$$

Not only is this function tractable, but it also has the following important property

$$
\begin{align*}
J(q)
&= \sum_{\mathcal Z} q(\mathbf x)\log\frac{q(\mathbf x)}{\tilde{p}(\mathbf x)} \\
&= \sum_{\mathcal Z} q(\mathbf x)\log\frac{q(\mathbf x)}{p(\mathbf x)}-\log Z \\
&= \text{KL}(q\|p)-\log Z
\end{align*}
$$

Since $Z$ is a constant, by minimizing $J(q)$, we will force $q$ to become close to $p$. Additionally, since KL divergence is always non-negative, we see that $J(q)$ is an upper bound on the negative log-likelihood (NLL):

$$J(q)= \text{KL}(q\|p)-\log Z\geq  -\log Z= -\log p(\mathcal D)$$

Alternativey, we can maximize the _energy functional_ $\mathcal L(q)$, which is a lower bound of the log-likelihood of the data

$$
\mathcal L(q)\triangleq-J(q)=-\text{KL}(q\|p)+\log Z\leq \log Z = \log p(\mathcal D)
$$

Therefore, $\mathcal L(q)$ is also called the evidence lower bound (ELBO) or the variational lower bound.

### Alternative interpretations of the variational objective

In statistical physics, $J(q)$ is called the _variational free energy_ or the _Hellmholtz free energy_, and is equal to the expected energy minus the entropy of the system

$$
J(q)=\mathbb E_q[\log q(x)] + \mathbb E_q[-\log\tilde p(x)]=-\mathbb H(q)+\mathbb E_q[E(x)]
$$

Another formuation of $J(q)$ as the expected NLL plus a penalty term that measures how far the approximate posterior is from the exact prior

$$
\begin{align*}
J(q)
&=\mathbb E_q[\log q(\mathbf x)-\log p(\mathbf x)p(\mathcal D|\mathbf x)] \\
&=\mathbb E_q[\log q(\mathbf x)-\log p(\mathbf x) - \log p(\mathcal D|\mathbf x)] \\
&=\mathbb E_q[-\log p(\mathcal D|\mathbf x)]+\text{KL}(q\|p)
\end{align*}
$$

### Forwards or reverse KL?

$\text{KL}(q\|p)\neq\text{KL}(p\|q)$ and both divergences equal zero when $q=p$, but assign different penalties when $q\neq p$.

Computational-wise, optimizing $\text{KL}(q\|p)$ involves an expectation w.r.t. $q$, while $\text{KL}(p\|q)$ requires computing expecatations w.r.t $p$, which is typically intractable to even evaluate. The choice of divergence also affects the returned solution when the approximating family $\mathcal Q$ does not contain the true $p$.

First, consider the reverse KL, $\text{KL}(q\|p)$, aka an I-projection or information projection,

$$\text{KL}(q\|p)=\sum_xq(\mathbf x)\log\frac{q(\mathbf x)}{p(\mathbf x)}$$

This is is infinite if $p(x)=0$ and $q(x)>0$. Thus if $p(\mathbf x)=0$ we must ensure $q(\mathbf x)=0$. We say that $\text{KL}(q\|p)$ is _zero-forcing_ for $q$ and it will typically under-estimate the support of $p$.

Now consider the forwards KL, $\text{KL}(p\|q)$, aka an M-projection or momemt project,

$$\text{KL}(p\|q)=\sum_xp(\mathbf x)\log\frac{p(\mathbf x)}{q(\mathbf x)}$$

This is infinite if $q(\mathbf x)=0$ and $p(\mathbf x)>0$. Thus, if $p(\mathbf x)>0$ we must ensure $q(\mathbf x)>0$. We say that $\text{KL}(p\|q)$ is z*ero-avoiding* for $q$ and it will typically over-estimate the support of $p$.

The difference between these methods is illustrated in the figure below, where $p$ is the blue contours and $q$ red. We see that when the true distribution is multimodal, using the forwards KL is a bad idea (assuming $q$ is unimodal), since the resulting poster mode/mean will be in a region of low density, right between the two peaks.

![KL divergences](kl-divergences.png)

## The mean-field method

The next step in our development of variational inference concerns the choice of approximating family $\mathcal Q$. One of the most popular forms is called the _mean-field_ approximation. In this approach, we assume the posterior is a fully factorized approximation of the form

$$q(\mathbf x)=\prod_i q_i(\mathbf x_i)$$

Our goal is to solve this optimization problem:

$$
\min_{q_1, ..., q_n} J(q)\equiv \max_{q_1, ..., q_n} \mathcal L(q)
$$

where we optimize over the parameters of each marginal distribution $q_i$.

The standard way of performing this optimization problem is via coordinate descent over the $q_j$. We iterate over $j=1, 2, ..., n$ and for each $j$ we optimize $\text{KL}(q\|p)$ over $q_j$ while keeping the other "coordinates" $q_{-j}=\prod_{i\neq j}q_i$ fixed. This has a simple closed form solution:

$$
\log q_j(x_j)\leftarrow\mathbb E_{q_{-j}}[\log\tilde p(x)]+\text{const}
$$

Notice that:

- Both sides of the above equation contain univariate functions of $\mathbf x_j$: we are thus replacing $q(x_j)$ with another function of the same form. The constant term is a normalization constant for the new distribution.
- On the right-hand side, we are taking an expectation of a sum of factors $\log\tilde p(\mathbf x)=\sum_k\log\phi(\mathbf x_k)$. Only factors belonging to the Markov blanket of $\mathbf x_j$ are a function of $x_j$; the rest are constant w.r.t $\mathbf x_j$ and can be pushed to the constant term. Sicne we are replacing the neigghbouring values by their mean value, the method is known is mean field.
- Tis leaves us with an expectation over a much smaller number of factors.

The result of this is a procedure that iteratively fits a fully-factored $q(\mathbf x)=\prod_ip_i(\mathbf x_i)$ that approximates $p$ in terms of $\text{KL}(q\|p)$. After each step of coordinate descent, we increase the variational lower bound, tightening it around $\log Z$. In the end, the factors $q_j(\mathbf x_j)$ will not quite equal the true marginal distributions $p(\mathbf x_j)$, but they will often be good enough for many practical purposes, such as determining $\max_{\mathbf x_j} p(\mathbf x_j)$.

### Derivation of the mean-field update equations

We will maximize $\mathcal L(q)$ w.r.t. one $q_j$ term at a time. Single out the terms that involve $q_j$ and regard all the other terms as constants, we get

$$
\begin{align*}
\mathcal L(q_j)
&= \sum_{\mathcal Z}\prod_iq_i(\mathbf x_i)\left[\log\tilde p(\mathbf x)-\sum_k\log q_k(\mathbf x_k)\right] \\
&= \sum_{\mathbf x_j}\sum_{\mathbf x_{-j}}q_j(\mathbf x_j)\prod_{i\neq j}q_i(\mathbf x_i)\left[\log\tilde p(\mathbf x)-\sum_k\log q_k(\mathbf x_k)\right] \\
&= \sum_{\mathbf x_j}q_j(\mathbf x_j)\sum_{\mathbf x_{-j}}\prod_{i\neq j}q_i(\mathbf x_i)\log\tilde p(\mathbf x) - \sum_{\mathbf x_j}q_j(\mathbf x_j)\sum_{\mathbf x_{-j}}\prod_{i\neq j}q_i\left[\log q_j(\mathbf x_j)+\sum_{k\neq j}\log q_k(\mathbf x_k)\right] \\
&= \sum_{\mathbf x_j}q_j\log f_j(\mathbf x_j) - \sum_{\mathbf x_j}q_j\log q_j(\mathbf x_j) + \text{const}
\end{align*}
$$

where

$$
\log f_j(\mathbf x_j)\triangleq\sum_{\mathbf x_{-j}}\prod_{i\neq j}q_i(\mathbf x_i)\log\tilde p(\mathbf x)=\mathbb E_{q_{-j}}[\log\tilde p(\mathbf x)]
$$

So we average out all the hidden variables except for $\mathbf x_j$. We can thus rewrite $\mathcal L(q_j)$ as follows

$$
\mathcal L(q_j) = -\text{KL}(q_j\|f_j)
$$

We can maximize $\mathcal L$ by minimizing this KL, which we can do by setting $q_j=f_j$ as follows

$$
\begin{align*}
q_j(\mathbf x_j)&=\frac 1 {Z_j}\exp(\mathbf E_{q_{-j}}[\log\tilde p(\mathbf x)])
\\
\log q_j(\mathbf x_j)&=\mathbf E_{q_{-j}}[\log\tilde p(\mathbf x)]-\log Z_j
\end{align*}
$$

The additive constant $Z_j$ is set by normalizing $\exp(\mathbf E_{q_{-j}}[\log\tilde p(\mathbf x)])$.

The functional form of the $q_j$ distributions will be determined by the type of variables $\mathbf x_j$, as well as the form of the model. This is sometimes called free-form optimization. If $z_j$ is a discrete r.v., then $q_j$ will be a discrete distributions; if $\mathbf x_j$ is a continuous random variable, then $q_j$ will be some kind of pdf.

### Ising model

### Univariate Gaussian: Variational Bayes (VB)

### Mixtures of Gaussians: Variational Bayes EM (VBEM)

## Loopy belief propagation (LBP)

LBP is a very simple approximate inference algorithm for discrete (or Gaussian) graphical models. The basic idea: we apply the belief propagation algorithm to the graph, even if it has loops (i.e., even if it is not a tree).

The algo below outlines LBP for pairwise models. To handle models with higher-order clique potentials (which include directed models where some nodes have more than one parent), we simply apply LBP on the factor graph.

Note that convergence is not guaranteed.

---

**LBP for a pairwise MRF**

1. Input: node potentials $\psi_s(x_s)$, edge potentials $\psi_{st}(x_s, x_t)$.
2. Initialize messages $m_{s\rightarrow t(x_t)}=1$ for all edges $s-t$.
3. Initialize beliefs $\text{bel}_s(x_s)=1$ for all nodes $s$.
4. Repeat until beliefs don't change significantly:
   - Send message on each edge
   - $m_{s\rightarrow t}(x_t)=\sum_{x_s}\left(\psi_s(x_s)\psi_{st}(x_s, x_t)\prod_{u\in N(s)\setminus t}m_{u\rightarrow s}(x_s)\right)$.
   - Update belief of each node $\text{bel}_s(x_s)\propto\psi_s(x_s)\prod_{t\in N(s)}m_{t\rightarrow s}(x_s)$
5. Return maginal beliefs $\text{bel}_s(x_s)$

---

### Convergence of LBP

The computation tree visualizes the messages that are passed as the algorithm proceeds. The key insight is that $T$ iterations of LBP is equivalent to exact computation in a computation tree of height $T+1$. If the strengths of the connections on the edges is sufficiently weak, then the influence of the leaves on the root will diminish over time, and convergence will occur.

Below is a figure of (a) a simple loopy graph and (b) its computation tree.

![A simple loopy graph and its computation tree](lbp-computation-tree.png)

### Making LBP converge

One simple way to reduce the chacne of oscillation is to use damping i.e. insteand of sending the messages $M_{ts}^k$, we send a damped messaged of the form

$$
\tilde M_{ts}^k(x_s)=\lambda M_{ts}(x_s)+(1-\lambda)\tilde M_{ts}^{k-1}(x_s)
$$

where $0\leq\lambda\leq 1$ is the damping factor. If $\lambda=1$ this reduces to the standard scheme, but for $\lambda<1$, this partial updating scheme can help with convergence. Using a value such as $\lambda\sim 0.5$ is a standard pracctice.

### Increasing the convergence rate: message scheduling

The standard approach when implementing LBP is to perform _synchronyous updates_, where all nodes absorb messages in parallel, and then send out messages in parallel i.e. the new messages at iteration $k+1$ are computed in parallel using

$$
\mathbf m^{k+1}=(f_1(\mathbf m^k), ..., f_E(\mathbf m^k))
$$

where $E$ is the number of edges and $f_{st}(\mathbf m)$ is the function that computes the message for edge $s\rightarrow t$ given all the old messages. This is analogous to the Jacobi method for solving linear systems of equations.

It is well known that the Gauss-Seidel method, which performs asynchronous updates in a fixed round-robin fashion, converges faster when solving linear systems of equations. We can apply the same idea to LBP, using updates of the form

$$
\mathbf m_k^{k+1}=f_i(\{\mathbf m_j^{k+1}|j<i\}, \{\mathbf m_j^k|j>i\})
$$

where the message for edge $i$ is computed using new messages (iteration $k+1$) from edges earlier in the ordering, and using old messages (iteration $k$) from edges later in the ordering.

## Reference materials

- Murphy, K. P. "More Variational Inference." _Machine Learning: A Probabilistic Perspective_.
- Murphy, K. P. "Variational Inference." _Machine Learning: A Probabilistic Perspective_.
- Kuleshov, V. and Ermon, S. "Variational inference." _cs228-notes_. Accessed Nov 6, 2021. https://ermongroup.github.io/cs228-notes/inference/variational/.
