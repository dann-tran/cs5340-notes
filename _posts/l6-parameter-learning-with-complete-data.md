---
title: Parameter Learning with Complete Data
tag: learning
lectureNumber: 6
---

## Learning methods

### Maximum likelihood estimate (MLE)

$$
\begin{align*}
\hat\theta
&\triangleq \argmax_{\theta}p(\mathcal D|\theta) \\
&= \argmax_{\theta}\log p(\mathcal D|\theta)
\end{align*}
$$

### Maximum a posteriori (MAP) estiamte

$$
\begin{align*}
\hat\theta
&\triangleq \argmax_{\theta}p(\theta|\mathcal D) \\
&= \argmax_{\theta}p(\mathcal D|\theta)p(\theta) \\
&= \argmax_{\theta}[\log p(\mathcal D|\theta)+\log p(\theta)]
\end{align*}
$$

Note that under strong sampling assumption, $p(\mathcal D|\theta)=\left[\frac 1 {\vert\theta\vert}\right]^I$. Since the likelihood $p(\mathcal D|\theta)$ depends exponentially on N, and the prior $p(\theta)$ stays constant, as we get more and more data, the MAP estimate converges towards the MLE. In other words, if we have enoughd ata, the data overwhelms the prior.

### The Bayesian approach

Whereas MLE and MAP give point estimates of $\theta$, the Bayesian approach uses the posterior $p(\theta|\mathcal D)=\frac{p(\mathcal D|\theta)p(\theta)}{p(\mathcal D)}$ to evaluate the predictive distribution i.e. admit many values of the parameters compatible with the data and weigh them according to the posterior density.

$$
p(\mathbf x|\mathcal D)=\int p(\mathbf x|\theta)p(\theta|\mathcal D)d\theta
$$

From the Bayesian approach, the posteriors for MAP and MLE can be considered as delta functions centered at their estimates $\hat\theta$ i.e. $p(\theta|\mathcal D)=\delta(\theta-\hat\theta)$. Then,

$$
p(\mathbf x|\mathcal D)=\int p(\mathbf x|\theta)\delta(\theta-\hat\theta)d\theta=p(\mathbf x|\hat\theta)
$$

## Single r.v. DGM

### Univariate normal distribution

### Univariate categoricacl distribution

## Directed models

Let $G=(U, E)$ be a directed graph where $U$ is the set of nodes and $E$ the set of edges. We associate a random vector $X$ with the graph, where the components of the vector are indexed by the nodes in the graph.

Probability model for a directed graph:

$$
p(x_U|\theta)=\prod_{u\in U}p(x_u|x_{\pi_u}, \theta_u)
$$

Construct the augmented graphical model for $N$ i.i.d. samples $G^{(N)}=(U^{(N)}, E^{(N)})$. The observed data will be $\mathcal D=(x_{U, 1}, x_{U, 2}, ..., x_{U, N})$.

Probability model for $G^{(N)}$ and the log-likelihood:

$$
\begin{align*}
p(\mathcal D|\theta)
&= \prod_n p(x_{U, n}|\theta) \\
&= \prod_n\prod_u p(x_{u, n}|x_{\pi_u, n}, \theta_u) \\
\ell(\theta; \mathcal D)&=\sum_n\sum_u\log p(x_{u, n}|x_{\pi_u, n}, \theta_u)
\end{align*}
$$

Note that the local subset of observations $\{x_{u, n}, x_{\pi_u, n}\}_{n=1}^N$, that is data associated with node $u$ and its parents, is sufficient for $\theta_u$.

### Directed, discrete models

Let $m(x_U)$ denote the number of times that $x_U$ is observed among the observations in the dataset $\mathcal D$. Define marginal counts $m(x_C)$ associated with subsets of nodes $C$ as the number of times that configuration $x_C$ is observed in the dataset. A particular subset of interest is the subset consisting of a node $u$ and its parents $\pi_u$ — the family associated with node $u$, denoted as $\phi_u\triangleq\{u\}\cup\pi_u$. We have:

$$
\begin{align*}
m(x_U)&\triangleq \sum_n \delta(x_U-x_{U, n}) \\
m(x_C)&\triangleq\sum_{x_U\setminus C}m(x_U) \\
m(x_{\phi_u})&\triangleq\sum_{x_{U\setminus\phi_u}}m(x_U)
\end{align*}
$$

Define the parameter vector $\theta_v({x_{\phi_v}})$ to be a nonnegative, multidimensional table indexed by the joint configuration of $v$ and $\pi_v$. The normalisation condition requires:

$$
\sum_{x_v}\theta_v(x_{\phi_v})=\sum_{x_v}\theta_v(x_v, x_{\pi_v})=1
$$

Define the local conditional probability of node $v$ using the normalized table:

$$
p(x_v|x_{\pi(v)}, \theta_v)\triangleq\theta_v(x_{\phi_v})
$$

Taking the product over $v$, we obtain the joint probability distribution as the product of normalized potentials:

$$
\begin{align*}
p(x_U|\theta)
&=\prod_v p(x_v|x_{\pi_v}, \theta_v) \\
&=\prod_v\theta_v(x_{\phi_v})
\end{align*}
$$

We take a further product over $n$ to obtain the total probability of an i.i.d. dataset $\mathcal D$:

$$
\begin{align*}
p(x_{U, n}|\theta)&=\prod_{x_U}p(x_U|\theta)^{\delta(x_U-x_{U, n})} \\
\log p(\mathcal D|\theta)
&= \log\prod_n p(x_{U, n}|\theta) \\
&= \sum_n\sum_{x_U}\delta(x_U-x_{U, n})\log p(x_U|\theta)  \\
&= \sum_{x_U}m(x_U)\log p(x_U|\theta) &(1) \\
&= \sum_{x_U}m(x_U)\sum_v\log\theta_v(x_{\phi_v}) \\
&= \sum_v\sum_{x_{\phi_v}}\sum_{x_{U\setminus\phi_v}}m(x_U)\log \theta_v(x_{\phi_v}) \\
&= \sum_v\sum_{x_{\phi_v}}\log \theta_v(x_{\phi_v})\sum_{x_{U\setminus\phi_v}}m(x_U) \\
&= \sum_v\sum_{\phi_v}m(x_{\phi_v})\log \theta_v(x_{\phi_v}) &(2)
\end{align*}
$$

$(1)$ shows that the sum over $n$ has disappeared; we have in essence reduced our representation of joint probability from a function on $G^{(N)}$ to a function on $G$. $(2)$ expresses the log-likelihood as a sum of terms defined on the families ${\phi_v}$. Furthermore, the likelihood can be seen as a exponential family with $m(x_{\phi_v})$ as the sufficient statistics and $\log \theta_v(x_{\phi_v})$ as the natural parameters.

To estimate $\theta_v(x_{\phi_v})$, we maximize $m(x_{\phi_v})\log \theta_v(x_{\phi_v})$ w.r.t. $\theta_v(x_{\phi_v})$. Adding a Lagrangian term to handle the normalization constraint, we obtain

$$
\hat\theta_{v, \text{ML}}(x_{\phi_v})=\frac{m(x_{\phi_v})}{m(x_{\pi_v})}=\frac{m(x_v, x_{\pi_v})}{m(x_{\pi_v})}
$$

## Undirected models

Undirected graphical models require an explicit global normalization factor that couples the parameters and complicates the parameter estimation problem. However, for decomposable models, the parameter estimation problem decouples.

Parameterize an undirected graphical model via a set of clique potentials $\psi_C(x_C)$, for $C\in\mathcal C$ where $\mathcal C$ is a set of cliques. Define the joint probability as

$$
p(x_U|\theta)=\frac 1 Z\prod_C \psi_C(x_C)
$$

where $\theta=\{\psi_C(x_C), C\in\mathcal C\}$ is the collection of parameters, and where $Z$ is the normalization factor $Z=\sum_{x_U}\prod_C \psi_C(x_C)$.

Log-linear form:

$$
p(x_U|\theta)=\frac 1 Z\exp\left(\sum_C\theta_C^T\phi_C(x_C)\right)
$$

### Undirected, discrete models

$$
\begin{align*}
p(\mathcal D|\theta)&=\prod_np(x_{U, n}|\theta) \\
&=\prod_n\prod_{x_U}p(x_U|\theta)^{\delta(x_U-x_{U, n})} \\
\ell(\theta;\mathcal D)&=\log p(\mathcal D|\theta) \\
&=\sum_n\sum_{x_U}\delta(x_U-x_{U, n})\log p(x_U|\theta) \\
&=\sum_{x_U}m(x_U)\log p(x_U|\theta) \\
&=\sum_{x_U}m(x_U)\log\left(\frac 1 Z\prod_C \psi_C(x_C)\right) \\
&=\sum_{x_U}m(x_U)\sum_C\log\psi_C(x_C)-\sum_{x_U}m(x_U)\log Z \\
&=\sum_C\sum_{x_C}m(x_C)\log\psi_C(x_C)-N\log Z
\end{align*}
$$

We see that the marginal counts $m(x_C)$, for $C\in\mathcal C$, are the sufficient statistics for our modlel. This is reminiscent of the directed case, where the cliques $\mathcal C$ were the families $\{\phi_v\}$.

#### In log-linear form

We use scaled log-likelihood

$$
\ell(\theta; \mathcal D)\triangleq\frac 1 N\sum_n\log p(x_{U, n}|\theta)=\frac 1 N\sum_n\left[\sum_C \theta_C^T\phi_C(x_C)-\log Z\right]
$$

### MLE

The derivative of the first term w.r.t $\psi_C(x_C)$ is $\frac{m(x_C)}{\psi_c{x_C}}$. For the second term $\log Z$,

$$
\begin{align*}
\frac{\partial\log Z}{\partial\psi_C(x_C)}
&=\frac 1 Z \frac\partial{\partial\psi_C(x_C)}\sum_{\tilde x}\prod_{\mathcal D}\psi_{\mathcal D}(x_{\mathcal D}) \\
&=\frac 1 Z\sum_{\tilde x}\delta(\tilde x_C-x_C)\frac\partial{\partial\psi_C(x_C)}\prod_{\mathcal D}\psi_{\mathcal D}(\tilde x_{\mathcal D}) \\
&=\frac 1 Z\sum_{\tilde x}\delta(\tilde x_C-x_C)\prod_{\mathcal D\neq C}\psi_{\mathcal D}(\tilde x_{\mathcal D}) \\
&=\sum_{\tilde x}\delta(\tilde x_C-x_C)\frac 1{\psi_C(\tilde x_C)}\frac 1 Z \prod_{\mathcal D}\psi_{\mathcal D}(\tilde x_{\mathcal D}) \\
&=\frac 1 {\psi_C(x_C)}\sum_{\tilde x}\delta(\tilde x_C-x_C)p(\tilde x) \\
&=\frac{p(x_C)}{\psi_C(x_C)}
\end{align*}
$$

Therefore,

$$
\frac{\partial\ell}{\partial\psi_C(x_C)}=\frac{m(x_C)}{\psi_c(x_C)}-N\frac{p(x_C)}{\psi_C(x_C)}
$$

Assume WLOG that $\psi_C(x_C)>0$ and equate the derivatie to zero, we obtain:

$$
\hat p_{\text ML}(x_C)=\frac 1 N m(x_c)
$$

Define the empirical distribution $p_\text{emp}(x)\triangleq m(x)/N$ so that $p_\text{emp}(x_C)\triangleq m(x_C)/N$ is a marginal under the empirical distribution, we can rewrite the result as:

$$
\hat p_{\text{ML}}(x_C)=p_\text{emp}(x_C)
$$

Thus we have the following important characterization of MLEs: for each clique $C\in\mathcal C$, the model marginals must be equal to the empirical marginals. This forms a system of equations that constrains the MLEs.

- For decomposable models, use the recipe below.
- For general undirected graphs, use IPF or SGD

#### In log-linear form

$$
\frac{\partial\ell}{\partial\theta_C}=\mathbb E_{p_{\text{emp}}}[\phi_C(x_C)]-\mathbb E_{p(\cdot|\theta)}[\phi_C(x_C)]
$$

This first term is the _clamped term_ and the second _unclamped term_ or _contrastive term_. At the optimum, the gradient is zero, yielding _moment matching_ $\mathbb E_{p_{\text{emp}}}[\phi_C(x_C)]=\mathbb E_{p(\cdot|\theta)}[\phi_C(x_C)]$.

### Decomposable models

A graph is said to be decomposable if it can be recursively subdivided into disjoint sets $A$, $B$, and $S$ where $S$ separates $A$ and $B$, and where $S$ is complete.

We can find MLEs for decomposable grphs by inspection, but only if the potentials are defined on maximal cliques i.e. our parameterization must be s.t. the set $\mathcal C$ ranges over the maximal cliques in the graph. Given this constraint, the recipe is the following:

- for every clique $C$, set the clique potential to the empirical marginal for that clique,
- for every non-empty intersection between cliques, associate an empirical marginal with that intersection, and divide that empirical marginal into the potential of one of the two cliques that form the intersection.

Example, for the figure below, we would have the following MLE

![A decomposable graph](decomposable-graph.png)

$$
\hat p_{\text{ML}}(x_1, x_2, x_3, x_4)=\frac{p_\text{emp}(x_1, x_2, x_3)p_\text{emp}(x_2, x_3, x_4)}{p_\text{emp}(x_2, x_3)}
$$

### Iterative proportional fitting (IPF)

IPF converges and ascends an objective function at each step. It is both a fixed-point algorithm and a coordinate ascent algorithm.

#### IPF as fixed-point iteration

Let us return to the gradient of log-likelihood, but retain the $\psi_C(x_C)$ factors. We have

$$
\frac{p_\text{emp}(x_C)}{\psi_C(x_C)}=\frac{p(x_C)}{\psi_C(x_C)}
$$

Note that the parameter $\psi_c(x_C)$ appears explicitly in this equation in two places, but also appears implicitly in the marginal $p(x_C)$. We can obtain an iterative algo by holding the values of $\psi_C(x_C)$ fixed on the RHS, and solving for the free parameter $\psi_C(x_C)$ on the LHS.

$$\psi_C^{(t+1)}(x_C)=\psi_C^{(t)}(x_C)\frac{p_\text{emp}(x_C)}{p^{(t)}(x_C)}$$

---

**IPF for tabular MRFs**

1. Initialise $\phi_C=1$ for $C\in\mathcal C$.
2. Repeat until convergence, for $C\in\mathcal C$:
   - $p_C=p(x_C|\phi)$.
   - $\hat p_C=p_{\text{emp}}(x)$.
   - $\phi_C=\phi_C\times\frac{\hat p_C}{p_C}$

---

Properties of the IPF update equation:

- the marginal $p^{(t+1)}(x_C)$ is equal to the empirical marginal $p_\text{emp}(x_C)$, and
- the normalization factor $Z$ remains constant across IPF updates.

Proof:

$$
\begin{align*}
p^{(t+1)}(x_C)
&=\sum_{x_{U\setminus C}}p^{(t+1)}(x) \\
&=\sum_{x_{U\setminus C}}\frac 1 {Z^{(t+1)}}\prod_{\mathcal D}\psi_{\mathcal D}^{(t+1)}(x_{\mathcal D}) \\
&=\frac 1 {Z^{(t+1)}}\sum_{x_{U\setminus C}}\psi_C^{(t+1)}(x_C)\prod_{\mathcal D\neq C}\psi_{\mathcal D}^{(t)}(x_{\mathcal D}) \\
&=\frac 1 {Z^{(t+1)}}\sum_{x_{U\setminus C}}\psi_C^{(t)}(x_C)\frac{p_\text{emp}(x_C)}{p^{(t)}(x_C)}\prod_{\mathcal D\neq C}\psi_{\mathcal D}^{(t)}(x_{\mathcal D}) \\
&=\frac{Z^{(t)}}{Z^{(t+1)}}\frac{p_\text{emp}(x_C)}{p^{(t)}(x_C)}\sum_{x_{U\setminus C}}\frac 1{Z^{(t)}}\prod_{\mathcal D}\psi_{\mathcal D}^{(t)}(x_{\mathcal D}) \\
&=\frac{Z^{(t)}}{Z^{(t+1)}}\frac{p_\text{emp}(x_C)}{p^{(t)}(x_C)}p^{(t)}(x_C) \\
&=\frac{Z^{(t)}}{Z^{(t+1)}}p_\text{emp}(x_C)
\end{align*}
$$

Note that both $p^{(t+1)}(x_C)$ and $p_\text{emp}(x_C)$ are normalized. Thus, summing both sides of the above equations w.r.t $x_C$, we get $Z^{(t+1)}=Z^{(t)}$. This further implies that $p^{(t+1)}(x_C)=p_\text{emp}(x_C)$.

IPF in terms of joint probabilities:

$$
\begin{align*}
p^{(t+1)}(x_U)
&=p^{(t)}(x_U)\frac{p_\text{emp}(x_C)}{p^{(t)}(x_C)} \\
&=p^{(t)}(x_{U\setminus C}|x_C)p_\text{emp}(x_C)
\end{align*}
$$

Interpretation: IPF iteration rettains the "old" conditional probability $p^{(t)}(x_{U\setminus C}|x_C)$ while replacing the "old" marginal probability $p^{(t)}(x_C)$ with the new marginal $p_\text{emp}(x_C)$.

In the case of decomposable models, IPF converges in one iteration.

#### IPF as coordinate ascent

A "coordinate" in this setting is a potential function. Take the derivative of log-likelihood w.r.t the coordiante $\psi_C(x_C)$, fior fixed $C$ and varying $x_C$, and solve for the maximizing values of these parameters while holding the remaining potentials fixed.

From the earlier derivation,

$$
\frac{\partial\ell}{\partial \psi_C(x_C)}=
\frac{m(x_C)}{\psi_c{x_C}}-\frac N Z\sum_{\tilde x}\delta(\tilde x_C-x_C)\prod_{\mathcal D\neq C}\psi_{\mathcal D}(\tilde x_{\mathcal D})
$$

Take parameter $\psi_C(x_C)$ on the RHS as a variable whose maximizing value we wish to solve for, where the remaining parameters $\psi_{\mathcal D}(x_{\mathcal D})$ are held fixed. Since $Z$ remains constant during IPF updates,

$$
\begin{align*}
\frac{\partial\ell}{\partial \psi_C(x_C)}
&=\frac{m(x_C)}{\psi_c^{(t+1)}{x_C}}-\frac N Z\sum_{\tilde x}\delta(\tilde x_C-x_C)\prod_{\mathcal D\neq C}\psi_{\mathcal D}^{(t)}(\tilde x_{\mathcal D}) \\
&=\frac{m(x_C)}{\psi_c^{(t+1)}{x_C}}-\frac N{\psi_C^{(t)}(x_C)}\sum_{\tilde x}\delta(\tilde x_C-x_C)\frac 1 Z\prod_{\mathcal D}\psi_{\mathcal D}^{(t)}(\tilde x_{\mathcal D}) \\
&=\frac{m(x_C)}{\psi_c^{(t+1)}{x_C}}-\frac N{\psi_C^{(t)}(x_C)}p^{(t)}(x_C)
\end{align*}
$$

and we see that the IPF update equation $\psi_C^{(t+1)}(x_C)=\psi_C^{(t)}\frac{p_\text{emp}(x_C)}{p^{(t)}(x_C)}$ does indeed set the gradient of the log-likelihood to zero, and thus a coordinate ascent step.

### Gradient descent

$$
\psi_C^{(t+1)}(x_C)=\psi_C^{(t)}(x_C)+\frac\rho{\psi_C^{(t)}(x_C)}\left(p_\text{emp}(x_C)-p^{(t)}(x_C)\right)
$$

where $\rho$ is a step size. We see that the difference between the empirical marginals and the model marginals drives the algorithm.

Advantage: All parameters can be adjusted silmultaneously (although a variant of IPF can also achieve this).

Disadvantages:

- The need to choose a step size.
- The noramlization factor $Z$ does not remain constant and must be recalculated anew after each iteration.

### In log-linear form

$$
\triangledown_{\mathbb\theta}\ell(\mathbb\theta; \mathcal D)=\frac 1 N\sum_n[\bm\phi(x_{U,n})-\mathbb E[\bm\phi(x_U)]]
$$

We can approximate the model expectations using MC sampling.

---

**SGD ML for fitting an MRF**

1. initialise weights $\bm\theta$ randomly.
2. $k=0, \eta=1$.
3. for each epoch, for each minibatch of size $B$:
   - for each sample $s=1:S$, sample $x^{s, k}\sim p(x|\theta_k)$.
   - $\hat{\mathbb E}[\bm\phi(x)]=\frac 1 S\sum_s\bm\phi(x^{s, k})$.
   - for each training case $i$ in minibatch, $\mathbf g_{ik}=\bm\phi(x_i)-\hat{\mathbb E}[\bm\phi(x)]$.
   - $\mathbf g_k=\frac 1 B\sum_{i\in B}\mathbf g_{ik}$.
   - $\bm\theta_{k+1}=\bm\theta_k-\eta\mathbf g_{k}$.
   - $k = k+1$.
   - Decrease step size $\eta$.

---

## CRFs

## Reference materials

- Prince, S. J. D. (2012). Chapter 4: Fitting probability models. In _Computer Vision: Models, Learning, and Inference_ (pp. 28-43). Cambridge University Press.
- Murphy, K. P. (2012). Undirected graphcial models (Markov random fields). In _Machine Learning: A Probabilistic Perspective_ (pp. 676-684). The MIT Press.
- Jordan, M. I. (2002). Chapter 9: Completely Observed Graphical Models. In _An Introduction to Probabilistic Graphical Models_.
