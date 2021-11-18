---
title: Hidden Markov Models (HMMs)
tag: modelling
lectureNumber: 8
---

## The graphical model

Sequential data: datasets in which successive samples are no longer assumed to be independent.

### Markov models

For discrete time steps:

$$
p(x)=p(x_1)\prod_{t=2}^Tp(x_t|x_{t-1})
$$

If the transition probability $p(x_t|x_{t-1})$ is independent of time, then the chain is called homogenous, stationary, or time-invariant.

If $x_t$ is discrete, we can represent $p(x_t|x_{t-1})$ as a transition matrix $A$, where $A_{ij}=p(X_t=j|X_{t-1}=i)$. Each row of the matrix sums to one, $\sum_j A_{ij}=1$, so this is called a stochastic matrix.

### HMM

An HMM is a natural generalization of a mixture model, viewed as a "dynamical" mixture model, where we no longer assume that the states (i.e. mixture components) are chosen independently at each step, but that the choice of a state at a given step depends on the choice of the state at the previous step. Thus we augment the basic mixture model to include a matrix of transition probabilites linking the states at neighbouring steps.

A HMM consists of a discrete-time, discrete-state Markov chain, plus an observation model aka emission probability $p(\mathbf x_t|z_t)$. The joint distribution:

$$
p(\mathbf z_{1:T}, \mathbf x_{1:T}|\bm\theta)
=p(\mathbf z_{1:T}|\bm\theta)p(\mathbf x_{1:T}|z, \bm\theta)
=p(z_1|\bm\pi)\prod_{t=2}^Tp(z_t|z_{t-1}, A)\prod_{t=1}^Tp(\mathbf x_t|z_t, \bm\phi)
$$

where

- Parameters $\bm\theta=\{\bm\pi, A, \bm\phi\}$
- Initial state distribution $p(z_1=i)=\pi_i$
- Transition matrix $A_{i, j}=p(z_t=j|z_{t-1}=i)$
- Class-conditional or emission densities $p(\mathbf x_t|z_t=k, \bm\phi)$. E.g.
  - Discrete: $p(\mathbf x_t=l|z_t=k, \bm\phi)=B_{k, l}$ where $B$is an observation matrix.
  - Continuous: $p(\mathbf x_t|z_t=k, \bm\phi)=\mathcal N(\mathbf x_t|\bm\mu_k, \bm\Sigma_k)$

## Inference in HMMs

### Types of inference problems for temporal models

Filtering: computes the belief state $p(z_t|\mathbf x_{1:t})$ online as the data streams in.

Smoothing: computes $p(z_t|\mathbf x_{1:t})$ offline, given all the evidence.

Fixed-lag smoothing: computes $p(z_{t-\ell}|\mathbf x_{1:t})$ where $\ell>0$ is the lag. This is a compromise between online and offline estimation and gives better performance than filtering, but incurs a slight delay. By changing the size of the lag, one can trade off accuracy vs delay.

Prediction: computes $p(z_{t+h}|\mathbf x_{1:t})$ where $h>0$ is the prediction horizon.

MAP estimation: computes $\argmax_{\mathbf z_{1:T}}p(\mathbf z_{1:T}|\mathbf x_{1:T})$. This is known as Viterbi decoding in the context of HMMs.

Posterior samples: $\mathbf z_{1:T}\sim p(\mathbf z_{1:T}|\mathbf x_{1:T})$. This can be done when there is more than one plausiable interpretation of the data.

Probability of the eivdence: $p(\mathbf x_{1:T})=\sum_{\mathbf z_{1:T}}p(\mathbf z_{1:T}, \mathbf x_{1:T})$

![The main kinds of inference for state-space models](inference-state-space-models.png)

### The forwards algo (online)

Goal: compute filtered marginal $\bm\alpha_t=p(z_t|\mathbf x_{1:t})$ aka _filtered belief state_ at time $t$.

Prediction step: computes the one-step-ahead predictive density; this acts as the new prior for time $t$.

$$
p(z_t=j|\mathbf x_{1:t-1})=\sum_i p(z_t=j|z_{t-1}=i)p(z_{t-1}=i|\mathbf x_{1:t-1})
$$

Update step: absorved the observed data from time $t$ using Bayes rule

$$
\begin{align*}
\alpha_t(j)
&\triangleq p(z_t=j|\mathbf x_{1:t}) \\
&=\frac 1{Z_t}p(\mathbf x_t|z_t=j)p(z_t=j|\mathbf x_{1:t-1}) \\
&=\frac 1{Z_t}\psi_t(j)\sum_i\alpha_{t-1}(j)\psi(i, j)
\end{align*}
$$

where $Z_t\triangleq p(\mathbf x_t|\mathbf x_{1:t-1})=\sum_j \psi_t(j)\sum_i\alpha_{t-1}(j)\psi(i, j)$

In matrix vector notation:

$$
\bm\alpha_t\propto\bm\psi_t\odot(\bm\Psi^T\bm\alpha_{t-1})
$$

where $\psi_t(j)=p(\mathbf x_t|z_t=j)$ is the local evidence at time $t$, $\Psi(i, j)=p(p_t=j|z_{t-1}=i)$ is the transition matrix.

Base case:

$$
\alpha_1(j)=p(z_1=j|\mathbf x_1)=\psi_1(j)
$$

### The forward-backward algo (offline)

Goal: compute the smoothed posterior marginal $\gamma_t(j)\triangleq p(z_t=j|\mathbf x_{1:T})$.

$$
\begin{align*}
\gamma_t(j)
&=p(z_t=j|\mathbf x_{1:T}) \\
&\propto p(z_t=j, \mathbf x_{t+1:T}|\mathbf x_{1:t}) \\
&\propto p(z_t=j|\mathbf x_{1:t})p(\mathbf x_{t+1:T}|z_t=j) \\
&=\alpha_t(j)\beta_t(j)
\end{align*}
$$

where $\beta_j(j)\triangleq p(\mathbf x_{t+1:T}|z_t=j)$ is the conditional likelhiood of future evidence given the hidden state at time $t$ is $j$.

The forwards algo recursively computes the $\alpha$'s in a left-to-right fashion. We'll recursively compute the $\beta$'s in a right-to-left fashion.

$$
\begin{align*}
\beta_{t-1}(i)
&=p(\mathbf x_{t:T}|z_{t-1}=i) \\
&=\sum_j p(z_t=j, \mathbf x_{t:T}|z_{t-1}=i) \\
&=\sum_j p(\mathbf x_{t+1:T}|z_t=j)p(\mathbf x_t|z_t=j)p(z_t=j|z_{t-1}=i) \\
&=\sum_j \beta_t(j)\psi_t(j)\psi(i, j)
\end{align*}
$$

We can write the resulting equation in matrix-vector form as

$$
\bm\beta_{t-1}=\bm\Psi(\bm\psi_t\odot\bm\beta_t)
$$

The base case is

$$
\beta_T(i)=p(\mathbf x_{T+1:T}|z_T=i)=p(\empty|z_T=i)=1
$$

### Two-slice smoothed marginals

When we use EM for learning, we'll need to compute the expected number of transitions from state $i$ to state $j$:

$$
\begin{align*}
N_{ij}
&=\sum_{t=1}^{T-1}\mathbb E[\mathbb I(z_t=i, z_{t+1}=j)|\mathbb x_{1:T}] \\
&=\sum_{t=1}^Tp(z_t=i, z_{t+1}=j|\mathbb x_{1:T})
\end{align*}
$$

Define the (smoothed) two-slice marginal

$$
\begin{align*}
\xi_{t, t+1}(i, j)
&\triangleq p(z_t=i, z_{t+1}=j|\mathbb x_{1:T}) \\
&\propto p(z_t|\mathbf x_{1:t})p(z_{t+1}|z_t, \mathbf x_{t+1:T}) \\
&\propto p(z_t|\mathbf x_{1:t})p(\mathbf x_{t+1:T}|z_t, z_{t+1})p(z_{t+1}|z_t) \\
&\propto p(z_t|\mathbf x_{1:t})p(\mathbf x_{t+1}|z_{t+1})p(\mathbf x_{t+2:T}|z_{t+1})p(z_{t+1}|z_t) \\
&=\alpha_t(i)\psi_{t+1}(j)\beta_{t+1}(j)\psi(i, j)
\end{align*}
$$

In matrix-vector form:

$$
\bm\xi_{t, t+1}\propto\bm\Psi\odot(\bm\alpha_t(\bm\psi_{t+1}\odot\bm\beta_{t+1})^T)
$$

### Time and space complexity

A straightforward implementation of FB takes $O(K^2T)$ time since we must perform a $K\times K$ matrix multiplication at each step. If the transition matrix is sparse, we can reduce this substantially e.g. $O(TK)$ for a left-to-right transition matrix.

The expected sufficient statistics needed by EM are $\sum_t\xi_{t-1, t}(i, j)$ which takes constant space. However, to compute them, we need $O(KT)$ working space, since we must store $\{\alpha_t\}_{t=1}^T$ until we do the backwards pass. It is possible to devise a simple divide-and-conquer algo that reduces the space complexity from $O(KT)$ to $O(K\log T)$ at the cost of increasing the running time from $O(K^2T)$ to $o(K^2T\log T)$.

## The Viterbi algo

The Viterbi algo computes the most probable sequence of states in a chain-structured grpahical model i.e. MAP

$$
\mathbf z^*=\argmax_{\mathbf z_{1:T}}p(\mathbf z_{1:t}|\mathbf x_{1:T})
$$

This is equivalent to computing a shortest path through the trellis diagram where the nodes are possible states at each time step, and the node and edge weights are log-probabilities i.e. the weight of a path $(z_t)_{t=1}^T$ is given by

$$
\log\pi_1(z_1)+\log\psi_1(z_1)+\sum_{t=2}^T[\log\psi(z_{t-1}, z_t)+\log\psi_t(z_t)]
$$

We cannot simply replace the sum-operator in forwards-backwards (sum-product) with a max-operator. In general max-product can lead to incorrect results if there are multiple equally probably joint assignments. This is because each node breaks ties independently and hence may do so in a manner that is inconssitent with its neighbours.

The Viterbi algo uses max-product for the forward pass and a traceback procedure for backward pass to recover the most probable path through the trellis of states. Once $z_t$ pics its most probable state, the previous nodes condition on this event, and therefore they will break ties consistently.

Define the probability of ending up in state $j$ at time $t$, given that we take the most probable path.

$$
\delta_t(j)\triangleq\max_{z_1, ..., z_{t-1}} p(\mathbf z_{1:t-1}, z_t=j|\mathbf x_{1:t})
$$

The most probable path to state $j$ at time $t$ must consist of the most probable path to some other state $i$ at time $t-1$, followed by a transition from $i$ to $j$. Hence

$$
\delta_t(j)=\max_i \delta_{t-1}(i)\psi(i, j)\psi_t(j)
$$

We also keep track of the most likely previous state on the most probable path to $z_t=j$ for all $j$:

$$
a_t(j)=\argmax_i\delta_{t-1}(i)\psi(i, j)\psi_t(j)
$$

We initialize by setting $\delta_1(j)=\pi_j\psi_1(j)$ and terminate by computing the most probable final state $z_T^*$:

$$
z_T^*=\argmax_i\delta_T(i)
$$

We can then compute the most probable sequence of states using traceback:

$$
z_t^*=a_{t+1}(z_{t+1}^*)
$$

Numerical underflow: work in log domain. We can use

$$
\begin{align*}
\log\delta_t(j)
&\triangleq \max_{\mathbf z_{1:t-1}}\log p(\mathbf z_{1:t-1}, z_t=j|\mathbf x_{1:t}) \\
&=\max_i\log\delta_{t-1}(i)+\log\psi(i, j)+\log\psi_t(j)
\end{align*}
$$

### The sum-product algo for HMM

## Learning: EM for HMMs (the Baum-Welch algo)

Consider $N$ i.i.d. replicates.

### E-step

$$
\mathcal Q(\bm\theta, \bm\theta^{\text{old}})=\sum_{k=1}^K\mathbb E[N_k^1]\log\pi_k+\sum_{j=1}^K\sum_{k=1}^K\mathbb E[N_{jk}]\log A_{jk}+\sum_{k=1}^KE[N_j]\log p(\mathbf x_{i, t}|\bm\phi_k)
$$

where the expected counts, computed using $\bm\theta^{\text{old}}$, are given by

$$
\begin{align*}
\mathbb E[N_k^1]&=\sum_{i=1}^N\gamma_{i, 1}(j) \\
\mathbb E[N_{jk}]&=\sum_{i=1}^N\sum_{t=2}^{T_i}\xi_{i, t}(j, k) \\
\mathbb E[N_j]&=\sum_{i=1}^N\sum_{t=1}^{T_i}\gamma_{i, t}(j)
\end{align*}
$$

### M-step

$$
\hat A_{jk}=\frac{\mathbb E[N_{jk}]}{\sum_{k'}\mathbb E[N_{jk'}]}, \space
\hat\pi_k=\frac{\mathbb E[N_k^1]}{N}
$$

#### Multinoulli observation model

The expected sufficient statistics are

$$
\begin{align*}
\mathbb E[M_{jl}]
&=\sum_{i=1}^N\sum_{t=1}^{T_i}\gamma_{i, t}(j)\mathbb I(x_{i, t}=l) \\
&=\sum_{i=1}^N\sum_{t:x_{i, t}=l}\gamma_{i, t}(j)
\end{align*}
$$

The M-step has the form

$$
\hat B_{jl}=\frac{\mathbf E[M_{jl}]}{\mathbf E[N_j]}
$$

#### Gaussian observation model

The expected sufficient statistics are

$$
\begin{align*}
\mathbb E[\overline{\mathbf x}_k] &= \sum_{i=1}^N\sum_{t=1}^{T_i}\gamma_{i, t}(k)\mathbf x_{i, t} \\
\mathbb E[(\overline{\mathbf x\mathbf x})_k^T] &= \sum_{i=1}^N\sum_{t=1}^{T_i}\gamma_{i, t}(k)\mathbf x_{i, t}\mathbf x_{i, t}^T \\
\end{align*}
$$

The M-step becomes

$$
\hat{\bm\mu}_k=\frac{\mathbb E[\overline{\mathbf x}_k]}{\mathbb E[N_k]}, \space
\hat{\bm\Sigma}_k=\frac
{\mathbb E[(\overline{\mathbf x\mathbf x})_k^T]-\mathbb E[N_k]\hat{\bm\mu}_k\hat{\bm\mu}_k^T}
{\mathbb E[N_k]}
$$

## Reference materials

- Murphy, K. P. (2012). Chapter 17: Markov and Hidden Markov Models. In _Machine Learning: A Probabilistic Perspective_. The MIT Press.
