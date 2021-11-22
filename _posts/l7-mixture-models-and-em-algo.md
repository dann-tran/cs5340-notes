---
title: Mixture Models and the EM Algorithm
tag: learning
lectureNumber: 7
---

## Latent variable models (LVMs)

An LVM $p(\mathbf x, \mathbf z|\bm\theta)$ is a probability distribution over two sets of variables $\mathbf x, \mathbf z$ where:

- variables $\mathbf x_n\in\mathbb{R}^d$ are observed at learning time in a dataset $\mathcal{D}=\{\mathbf x_1, ..., \mathbf x_N\}$, and
- latent/hidden variables $\mathbf z_n$ are never observed.

### Example: Gaussian mixture models (GMMs)

Our directed model: $p(\mathbf x, z|\bm\theta)=p(\mathbf x|z, \bm\theta)p(z|\bm\theta)$. Here $z$ is a scalar.

- $p(z|\bm\theta)=\text{Cat}(z|\bm\pi)$. Specifically, $\bm\pi$ is a probability vector where $p(z=k)=\pi_k$.
- $p(\mathbf x|z=k, \bm\theta)=\mathcal{N}(\mathbf x|\bm\mu_k, \bm\Sigma_k)$.

This model postulates that our observed data is comprised of $K$ clusters with proportions specified by $\pi_1, ...\pi_K$; the distribution within each cluster is a Gaussian. We can see that $p(\mathbf x)$ is a mixture by explicitly writing

$$
p(\mathbf x)=\sum_{k=1}^Kp(\mathbf x|z=k)p(z=k)=\sum_{k=1}^K\pi_k\mathcal{N}(\mathbf x|\bm\mu_k, \bm\Sigma_k)
$$

To generate a new data point, we sample a cluster $k$ and then sample its Gaussian $\mathcal{N}(\mathbf x|\bm\mu_k, \bm\Sigma_k)$.

![Gaussian mixture model](gmm.png)

### Reasons for modelling latent variables

- Some data might be naturally unobserved. E.g. patients dropping out during a clinical trials.

- LVMs enable us to leverage our prior knowledge when defining a model. E.g. topic modelling of news articles as a mixture of $K$ distinct distributions (one for each topic).

- LVMs increase the expressive power of our model.

### Marginal likelihood training

Our goal is still to fit the marginal likelihood $p(\mathbf X)$. Applying the argument for KL divergence, we should maximize the _observed data log-likelihood_

$$
\ell(\bm\theta)=\log p(\mathcal D|\bm\theta)=\sum_{\mathbf x\in \mathcal D}\log p(\mathbf x|\bm\theta)=\sum_{\mathbf x\in \mathcal D}\log\left(\sum_{\mathbf z} p(\mathbf x|\mathbf z, \bm\theta)p(\mathbf z)\right)
$$

This optimization objective is considerably more difficult than regular log-likelihood, even for directed graphical models. Reasons:

- The summation inside the log makes it impossible to decompose $p(\mathbf x)$ into a sum of log-factors. Even if the model is directed, we can no longer derive a simple closed form expression for the parameters.
- Whereas a single exponential family distribution $p(\mathbf x)$ has a concave log-likelihood, the log of a weighted mixture of such distributions is no longer concave or convex (contrast with complete data log-likelihood $\ell_c(\bm\theta)$ which is concave). This non-convexity requires the development of specialized learning algorithms.

![Non-convexity of mixture models](mixture-nonconvex.png)

## Learning LVMs

### The Expectation-Maximization (EM) algo

Consider the _complete data log-likelihood_

$$
\ell_c(\bm\theta)
\triangleq \log p(\mathbf X, \mathbf Z|\bm\theta)
= \sum_{i=1}^N\log p(\mathbf x_n, \mathbf z_n|\bm\theta)
$$

If latent $\mathbf Z$ were fully observed, we could compute $\ell_c(\bm\theta)$ and optimize it. However, since $\mathbf Z$ is unknown:

- we'll instead rely on a "soft" assignment of $\mathbf Z$ in the form of a posterior $p(\textbf Z|\textbf X, \bm\theta)$ to optimize the log-likelihood w.r.t. the parameters;
- the posterior $p(\textbf Z|\textbf X, \bm\theta)$ can often be efficiently computed if the parameters are known (an assumption, not ture for some models).

Our resulting tractable objective, _auxillary function_ $\mathcal Q$, is the _expected complete data log-likelihood_ under latent variable distribution:

$$
\mathcal{Q}(\bm\theta, \bm\theta^{\text{old}})=\mathbb{E}_{\textbf Z\sim p(\textbf Z|\textbf X, \bm\theta^{\text{old}})}[\ell_c(\bm\theta)]
$$

EM follows a simple iterative two-step strategy:

- Given an estimate $\bm\theta^{\text{old}}$ of the parameters, compute posterior $p(\textbf Z|\textbf X, \bm\theta^{\text{old}})$ as the _expected sufficient statistics_ for our MLE. This can be seen as a "soft" assignment of $\mathbf x_n$ to $K$ clusters (a hard assignment would assign $x_n$ to a single cluster).
- Then, find a new estimate $\bm\theta^{\text{new}}$ by optimizing $\mathcal{Q}$ w.r.t. $\bm\theta$. This process will eventually converge.

---

**EM algo**

1. Start at an initial $\theta^{\text{old}}$.

2. _E-step_: Compute the posterior $p(\textbf Z|\textbf X, \bm\theta^{\text{old}})$

3. _M-step_: Compute the new paramters

$$
\bm\theta^{\text{new}}=\argmax_\theta\mathcal{Q}(\bm\theta, \bm\theta^{\text{old}})
$$

4. Check for convergence of either the log-likelihood or the parameter values. If not converged, $\bm\theta^{\text{old}}\leftarrow\bm\theta^{\text{new}}$ and return to step 2.

---

To perform MAP estimation, we modify the M-step as follows:

$$
\bm\theta^{\text{new}}=\argmax_{\bm\theta}\mathcal{Q}(\bm\theta, \bm\theta^{\text{old}}) + \log p(\bm\theta)
$$

### Example: GMMs

#### Auxiliary function

Here each data point is $(\mathbf x_n, z_n)$ and the complete data $(\mathbf X, \mathbf z)$. The expected complete data log-likelihood is given by

$$
\begin{align*}
\mathcal{Q}(\bm\theta, \bm\theta^{\text{old}})
&\triangleq \mathbb{E}_{\mathbf z\sim p(\mathbf z|\mathbf X, \bm\theta^{\text{old}})}\left[\sum_n\log p(\mathbf x_n, z_n|\bm\theta)\right] \\
&= \sum_n\mathbb{E}_{z_n\sim p(z_n|\mathbf x_n, \bm\theta^{\text{old}})}\left[\log\left[\prod_{k=1}^K\left(\pi_k p(\mathbf x|\bm\theta_k)\right)^{\mathbb{I}(z_n=k)}\right]\right] \\
&= \sum_n\sum_k\mathbb{E}_{z_n\sim p(z_n|\mathbf x_n, \bm\theta^{\text{old}})}[\mathbb{I}(z_n=k)]\log[\pi_k p(\mathbf x|\bm\theta_k)] \\
&= \sum_n\sum_k p(z_n=k|\mathbf x_n, \bm\theta^{\text{old}})\log[\pi_k p(\mathbf x|\bm\theta_k)] \\
&= \sum_n\sum_k r_{nk}\log \pi_k + \sum_n\sum_k r_{nk}\log p(\mathbf x_n|\bm\theta_k)
\end{align*}
$$

where $r_{nk}\triangleq p(z_n=k|\mathbf x_n, \bm\theta^{\text old})$ is the _responsibility_ that cluster _k_ takes for data point $i$.

#### E-step

We calculate $r_{nk}$ explicitly as

$$
\begin{align*}
r_{nk}
&= p(z_n=k|\mathbf x_n, \bm\theta^{\text old}) \\
&= \frac{p(z_n=k)p(\textbf x_n|z_n=k, \bm\theta^{\text old})}{\sum_{j=1}^K p(z_n=j)p(\textbf x_n|z_n=j, \bm\theta^{\text old})} \\
&= \frac{\pi_k\mathcal{N(\textbf x_n|\bm\mu_k, \bm\Sigma_k)}}{\sum_{j=1}^K \pi_j\mathcal{N(\textbf x_n|\bm\mu_j, \bm\Sigma_j)}}
\end{align*}
$$

#### M-step

We optimize $\mathcal{Q}$ w.r.t $\bm\theta=\{\pi_k, \bm\mu_k, \bm\Sigma_k\}_{k=1}^K$. For $\bm\pi$, using the method of Lagrange multiplier for the constraint $\sum_k\pi_k=1$ which requires us to u, we have

$$
\pi_k=\frac 1 N\sum_n r_{nk}=\frac {r_k} N
$$

where $r_k\triangleq \sum_n r_{nk}$ is the weighted number of points assigned to cluster $k$.

For $\bm\mu_k$ and $\bm\Sigma_k$, consider the parts of $\mathcal{Q}$ that depend on $\bm\mu_k$ and $\bm\Sigma_k$. We have

$$
\begin{align*}
\ell(\bm\mu_k, \bm\Sigma_k)
&=\sum_n\sum_k r_{nk}\log p(\mathbf x_n|\bm\theta_k) \\
&= -\frac 1 2 \sum_n r_{nk}\left[\log\vert\bm\Sigma_k\vert + (\mathbf x_n - \bm\mu_k)^T\bm\Sigma_k^{-1}(\mathbf x_n - \bm\mu_k)\right]
\end{align*}
$$

The new parameter estimates are thus given by

$$
\begin{align*}
\bm\mu_k&=\frac{\sum_n r_{nk}\mathbf x_i}{r_k} \\
\bm\Sigma_k&=\frac{\sum_n r_{nk}(\mathbf x_n-\bm\mu_k)(\mathbf x_n - \bm\mu_k)^T}{r_k} = \frac{\sum_n r_{nk}\mathbf x_n\mathbf x_n^T}{r_k} - \bm\mu_k\bm\mu_k^T
\end{align*}
$$

These equations make intuitive sense: the mean of cluster $k$ is the weighted average of all points assigned to cluster $k$, and the covariance is proportional to the weighted empirical scatter matrix.

### EM as variational inference

TBC

### Properties of EM

Convergence is guaranteed for EM because:

- The marginal likelihood increases after each EM iteration.
- The marginal likelihood is upper-bounded by its true global maximum

However, since the objective is non-convex, we have no gurantee to find the global optimum. In fact, EM in practice converges almost always to a local optimum, and moreover, that optimum heavily depends on the choice of initialization. Thus it is very common to use multiple restarts of the algorithm and choose the best one in the end.

## Reference materials

- Murphy, K. P. (2012). Chapter 11: Mixture Models and the EM Algorithm. In _Machine Learning: A Probabilistic Perspective_. The MIT Press.
- Kuleshov, V. and Ermon, S. _Learning in latent variable models_. CS228 notes. https://ermongroup.github.io/cs228-notes/learning/latent/.
