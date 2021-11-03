---
title: Monte Carlo Inference (Sampling)
---

Sampling methods can be used to perform intractable integrations c e.g. normalization, marginalization, expectation.

## The Monte Carlo principle

MC simulation draws an i.i.d. set of samples $\{x^{(i)}\}_{i=1}^N$ from a target density $p(x)$ defined on a high-dimenstional space $\mathcal{X}$. These $N$ samples can be used to approximate the target density with the empirical point-mass function

$$p(x)\approx\frac{1}{N}\sum_{i=1}^N\mathcal{1}_{x^{(i)}}(x),$$

where $\mathcal{1}_{x^{(i)}}(x)$ denotes indicator function at $x^{(i)}$.

Consequently, to approximate expectation $\mathbb{E}_{x\sim p}[f(x)]$ with tractable sums $I_N$ that converge,

$$\frac{1}{N}\sum_{i=1}^Nf(x^{(i)})=I_N\overset{\text{a.s.}}{\underset{N\rightarrow\infty}{\longrightarrow}}\mathbb{E}_{x\sim p}[f(x)]=\int_\mathcal{X}f(x)p(x)dx$$

- The estimate $I_N$ is unbiased, and by SLLN, it will almost surely (a.s.) to $I(f)$.
- If the variance of $\text{Var}_{x\sim p}[f(x)]<\infty$, then $\text{Var}_{x^i\overset{\text{i.i.d.}}{\sim}p}[I_N]=\frac{1}{N}\text{Var}_{x\sim p}[f(x)]$.
- CLT yields convergence in distribution of the error $\sqrt{N}(I_N-\mathbb{E}_{x\sim p}[f(x)])\underset{N\rightarrow\infty}{\implies}\mathcal{N}(0, \sigma_f^2)$.

## Standard distributions

To sample from $p(x)$ with a closed-form inverse CDF:

1. Sample $u\sim\mathcal{U}(0, 1)$.
2. $x=F_X^{-1}(u)$.

In the case of a multinomial distribution with $k$ possible outcomes and associated probabilities $\theta_1, ..., \theta_k$.

- Subdivide a unit interval into $k$ regions with region $i$ having size $\theta_i$.
- Sample uniformly from $[0, 1]$ and return the value of the region in which our sample falls.

![Reducing sampling from a multinomial distribution to sampling a uniform distribution in $[0, 1]$](multinomial-sampling.png)

## Forward Sampling (aka Ancestral Sampling)

### Sample from a prior

To sample from prior $p(x)=\prod_{i=1}^M p(x_i|{\bf \pi}_i)$ of a DGM, simply sample nodes in topological order, conditioned on the sampled values of the parent nodes.

"Forward sampling" can also be performed efficiently on undirected models if the model can be represented by a clique tree with a small number of variables per node (clique).

- Calibrate the clique tree, which gives us the marginal distribution over each node, and choose a node to be the root.
- To sample for each node:
  1. Marginalize over variables to get the marginal for a single variable $p(X_1|E=e)$.
  2. Sample $x_1\sim p(X_1|E=e)$ and incorporate $X_1=x_1$ as evidence.
  3. Proceed with sampling $x_2\sim p(X_2=x_2|X_1=x_1, E=e)$, $x_3\sim p(X_3=x_3|X_1=x_1, X_2=x_2, E=e)$ and so on.
- When moving down the tree to sample variables from other nodes, each node must send an updated message containing the values of the sampled variables.

### Sample from a posterior

Suppose $X=Z\cup E$. To sample from posterior $p(z|e)$, use forwards sampling on prior $p(x)$ and throw away all samples that are inconsistent with $e$ (i.e. logic sampling, which can be considered a special case of rejections sampling). Formally,

$$p(e)=\sum_zp(e,z)dz=\sum_xp(x)\mathbb{1}_e(x)dx=\mathbb{E}_{x\sim p}[\mathbb{1}_e(x)]\approx\frac{1}{N}\sum_{i=1}^N\mathbb{1}_e(x^{(i)})$$

$$p(z|e)=\frac{p(e,z)}{p(e)}\approx\frac{\frac{1}{N}\sum_{i=1}^{N}\mathbb{1}_{e,z}(x^{(i)})}{\frac{1}{N}\sum_{i=1}^N\mathbb{1}_{e}(x^{(i)})}=\frac{\sum \mathbb{1}_{e,z}(x^{(i)})}{\sum \mathbb{1}_{e}(x^{(i)})}$$

where $\mathbb{1}_e(x)=\begin{cases}1&\text{if }x\text{ is consistent with }e \\0&\text{otherwise}\end{cases}$.

Drawback: the overall probability of accepting a sample from the posterior decreases rapidly as the number of observed variables increases and as the number of states that those variables can take increases.

## Rejection sampling

Suppose that:

- Target distribution is $p(x)$ hard to sample directly from.
- $p(x)$ can be evaluated up to a normalising constant i.e. unormalized potential $\tilde{p}(x)$ can be readily evaluated, but $p(x)=\frac{1}{Z_p}\tilde{p}(x)$ is not ($Z_p$ is unknown).
- Proposal distribution $q(x)$ is easy to sample from.

Choose $M<\infty$ such that $\tilde{p}(x)\leq Mq(x)$.

---

**Rejection Sampling Algorithm**

Set $i=1$.

Repeat until $i=N$:

1. Sample $x^{(i)}\sim q(x)$ and $u\sim\mathcal{U}(0, 1)$.
2. If $u<\frac{\tilde{p}(x^{(i)})}{Mq(x^{(i)})}$ then accept $x^{(i)}$ and increment the counter $i$ by $1$. Otherwise, reject.

---

![Rejection sampling](rejection-sampling.png)

Limitations:

- It is not always possible to bound $p(x)/q(x)$ within a reasonable constant $M$ over the whole sapce $\mathcal{X}$.
- If $M$ is too large, the acceptance probability $\text{Pr}(x\text{ accepted})=\text{Pr}\left(u<\frac{\tilde{p}(x)}{Mq(x)}\right)\approx\frac{1}{M}$ will be too small.

## Importance sampling

Importance sampling takes all samples drawn from $q$ and reweigh them with _importance weights_. We must assume that the support of $q(x)$ includes the support of $p(x)$.

### Unnormalised importance sampling

When $p(x)$ can be evaluated, we can define $w(x)\triangleq p(x)/q(x)$. We thus have

$$
\begin{align*}
\mathbb{E}_{x \sim p}[f(x)]
&= \int f(x)p(x)dx \\
&= \int f(x) \frac{p(x)}{q(x)} q(x) dx \\
&= \mathbb{E}_{x\sim q}[f(x)w(x)]
\end{align*}
$$

The Monte Carlo estimate is

$$\hat{\mathbb{E}}_{x\sim p}[f(x)]=\frac{1}{N} \sum_{i=1}^N f(x^{(i)}) w(x^{(i)})$$

The variance of this new estimator is $\text{Var}[\hat{\mathbb{E}}_{x\sim p}[f(x)]]=\frac{1}{N}\text{Var}_{x \sim q}[f(x)w(x)]$ where

$$
\begin{align*}
\text{Var}_{x \sim q}[f(x)w(x)]
&= \mathbb{E}_{x\sim q}\left[\left(f(x)w(x)\right)^2\right] - \mathbb{E}_{x\sim q}[f(x)w(x)]^2 \\
&= \mathbb{E}_{x\sim q}\left[\left(f(x)w(x)\right)^2\right] - \mathbb{E}_{x\sim p}[f(x)]^2 \\
&\geq 0
\end{align*}
$$

To minimize the variance, we only need to minimize $\mathbb{E}_{x\sim q}\left[\left(f(x)w(x)\right)^2\right]$ because $\mathbb{E}_{x\sim p}[f(x)]^2$ does not depend on $q$. According to Jensen's inequality,

$$
\begin{align*}
\mathbb{E}_{x\sim q}\left[\left(f(x)w(x)\right)^2\right]
&\geq \mathbb{E}_{x\sim q}[\vert f(x)\vert w(x)]^2 \\
&= \left(\int \vert f(x)\vert p(x)dx\right)^2 \\
&= \mathbb{E}_{x\sim p}[\vert p(x)\vert]^2 \\
\end{align*}
$$

This lower bound can be attained by choosing _optimal importance distribution_

$$q^*(x)=\frac{\vert f(x)\vert p(x)}{\int{\vert f(x)\vert p(x)}}$$

If we can sample from this $q$ (and evaluate the corresponding weight), then we only need a single MC sample to compute the true value of our integral. However, sampling from such a $\tilde{q}(x)=\vert f(x)\vert p(x)$ is NP-hard in general. Nevertheless, this tells us that high sampling efficiency is achieved when we focus on sampling from $p(x)$ in the important regions where $\vert f(x)\vert p(x)$ is relatively large.

### Normalised importance sampling

Assume only $\tilde{p}(x)$ can be evaluated, we then define $w(x)\triangleq\tilde{p}(x)/q(x)$. We have

$$X_p=\int \tilde{p}(x)dx=\int \frac{\tilde{p}(x)}{q(x)}q(x)dx=\mathbb{E}_{x\sim q}[w(x)]$$

As a result,

$$
\begin{align*}
\mathbb{E}_{x\sim p}[f(x)]
&= \int f(x)p(x)dx \\
&= \frac{1}{X_p}\int f(x)\frac{\tilde{p}(x)}{q(x)}q(x)dx \\
&= \frac{1}{X_p}\mathbb{E}_{x\sim q}[f(x)w(x)] \\
&= \frac{\mathbb{E}_{x\sim q}[f(x)w(x)]}{\mathbb{E}_{x\sim q}[w(x)]} \\
&\approx \frac{\sum_{i=1}^Nf(x^{(i)})w(x^{(i)})}{\sum_{i=1}^Nw(x^{(i)})} \\
&= \sum_{i=1}^Nf(x^{(i)})\tilde{w}(x^{(i)})
\end{align*}
$$

where $\tilde{w}(x^{(i)})$ is the normalized importance weight.

Drawback: the normalized importance sampling estimator is biased. When $N=1$,

$$
\mathbb{E}_{x\sim q}\left[\hat{\mathbb{E}}_{x\sim p}[f(x)]\right]
=\mathbb{E}_{x\sim q}\left[\frac{f(x)w(x)}{w(x)}\right]
=\mathbb{E}_{x\sim q}[f(x)]\neq \mathbb{E}_{x\sim p}[f(x)]
$$

Fortunately, because the numerator and denominator are both unbiased, the normalized importance sampling estimator remains asymptotically unbiased.

### Example: Approximate posterior probability $p(x_i|e)$ (discrete space)

Denote posterior probabilities $p_e(z)\triangleq p(z|e)$ and $\tilde{p}_e(z)\triangleq p(z,e)$. We then have

$$
\begin{align*}
p(x_i|e)
&= p_e(x_i) \\
&= \sum_z\mathbb{1}_{x_i}(z)p_e(z) \\
&= \mathbb{E}_{z\sim p_e}[\mathbb{1}_{x_i}(z)]  \\
&\approx \sum_{i=1}^N\mathbb{1}_{x_i}(z)\tilde{w}(z^{(i)})
\end{align*}
$$

where the unnormalised importance weight is $w(z)\triangleq \tilde{p}_e(z)/q(z)=p(z,e)/q(z)$.

## Markov chain Monte Carlo (MCMC)

MCMC is a strategy for generating samples $x^{(i)}$ while exploring the state space $\mathcal{X}$ using a Markov chain mechanism. This mechanism is consturcted so that the chain spends more time in the most important regions, i.e. samples $x^{(i)}$ mimic samples drawn from target distribution $\tilde{p}(x)$.

### Markov Chain on finite state spaces

Consider a discrete-time stochastic process $x^{(i)}$ is called a Markov chain if it satisfies the _Markov assumption_:

$$p(x^{(i)}|x^{(i-1)}, ..., x^{1})=T(x^{(i)}|x^{(i-1)})$$

The chain is _homogenous_ if $T\triangleq T(x^{(i)}|x^{(i-1)})$ remains invariant for all $i$, with $\sum_{x^{(i)}}T(x^{(i)}|x^{(i-1)})=1$ for any $i$.

If the initial state $x^{(0)}$ is drawn from probability vector $p(x^{(0)})$, we may represent the probability $p(x^{(t)})$ of ending up in each state after $t$ steps as

$$p(x^{(t)})=T^tp(x^{(0)})$$

The _stationary distribution_ of the Markov chain is the limit $\pi=\lim_{t\rightarrow\infty}p(x^{(t)})$ if it exists.

### Existence of a stationary distribution

The high-level idea of MCMC will be to construct a Markov chain whose states will be joint assignments to the variables in the model and whose stationary distribution will equal the model probability $p$.

Two sufficient conditions on $T$ for finite-state Markov chain to have a stationary distribution:

- _Irreducibility_: It is possible to get from any state $x$ to any other state $x'$ with a positive probability in a finite number of steps. In other words, $T$ cannot be reduced to separate smaller matrices annd the transition graph is connected.
- _Aperiodicity_: The train should not be trapped in cycles. In other words, it is possible to return to any state at any time, i.e., there exists an $n$ such that for all $k$ and all $n'\geq n$, $p(x^{(n')}=k|x^{(n)}=k)>0$.

In the case of continuuous variables, the Markov chain must be _ergodic_.

A sufficient (but not necessary) condition that a particular distribution $p(x)$ is a stationary distribution is the following _detailed balance_ (reversibility) condition:

$$p(x^{(i)})T(x^{(i-1)}|x^{(i)})=p(x^{(i-1)})T(x^{(i)}|x^{(i-1)}) \enspace \forall x^{(i-1)}$$

Proof: summing both sides over $x^{(i-1)}$ gives us

$$p(x^{(i)})=\sum_{x^{(i-1)}}p(x^{(i-1)})T(x^{(i)}|x^{(i-1)})$$

MCMC samplers are irreducible and aperiodic Markov chains that have the target distribution as the invariant distribution. One way to design these samplers is to ensure that detailed balance is satisfied.

### MCMC algorithms

At a high level, MCMC algorithms will have the following structure. They take as argument a transition operator $T$ specifying a Markov chain whose stationary distribution is $p$ (unnormalised), and an initial assignment $x_0$ to the variables of $p$. An MCMC algorithm then perform the following steps.

1. Run the Markov chain from $x_0$ for $B$ _burn-in_ steps.
2. Run the Markov chain for $N$ _sampling_ steps and collect all the states that it visits.

Assuming $B$ is sufficiently large, the latter collection of states will form samples from $p$. We may then use these samples for Monte Carlo integration (or in importance sampling). We may also use them to:

- produce Monte Carlo estimates of marginal probabilities,
- perform MAP inference by take the sample with the highest probability

### Metropolis-Hastings algorithm

The Metropolis-Hastings (MH) algorithm is our first way to construct Markov chains within MCMC. The MH method constructs a transition operator $T$ from two components:

- A transition kernel $q(x^*|x)$, that is our proposal distribution.
- An acceptance probability for moving towards candidate value $x^*$ sampled from $q(x^*|x)$

$$\mathcal{A}(x^*, x)=\min\left\{1, \frac{p(x^*)q(x|x^*)}{p(x)q(x^*|x)}\right\}$$

---

**Rejection Sampling Algorithm**

1. Initialise $x^{(0)}$.
2. For $i=0$ to $N-1$
   - Sample $u\sim U(0, 1)$.
   - Sample $x^*\sim q(x^*|x^{(i)})$.
   - If $u<\mathcal{A}(x^*, x^{(i)})$, assign $x^{(i+1)}=x^*$; else, $x^{(i+1)}=x^{(i)}$

---

Notice that the acceptance probability encourages us to move towards more likely points in the distribution; when $q$ suggests that we move into a low-probability region, we follow that move only a certain fraction of the time.

In practice, the distribution $q$ is taken to be something simple, like a Gaussian centered at $x$ if we are dealing with continuous variables.

#### Fact 1: The MH algo admits $p$ as a stationary distribution.

This means that the MH algo will eventually produce samples from their stationary distribution, which is $p$ by construction.

We shall prove that $p$ satisfies the detailed balance condition w.r.t the MH Markov chain. The transition kernel for the MH algorithm is

$$T(x^{(i+1)}|x^{(i)})=q(x^{(i+1)}|x^{(i)})\mathcal{A}(x^{(i+1)}, x^{(i)})+\mathcal{1}_{x^{(i)}}(x^{(i+1)})r(x^{(i)})$$

where $r(x^{(i)})$ is the term associated with rejection

$$r(x^{(i)})=\int_{\mathcal{X}}q(x^*|x^{(i)})(1-\mathcal{A}(x^*, x^{(i)}))dx^*$$

Case 1: $x^{(i+1)}=x^{(i)}$ (candidate sample is rejected). Then the detailed balance condition is trivially evident.

Case 2: $x^{(i+1)}\neq x^{(i)}$. We have,

$$
\begin{align*}
T(x^{(i+1)}|x^{(i)})
&= q(x^{(i+1)}|x^{(i)})\min\left\{1, \frac{p(x^{(i+1)})q(x^{(i)}|x^{(i+1)})}{p(x^{(i)})q(x^{(i+1)}|x^{(i)})}\right\} \\
&= \frac{1}{p(x^{(i)})}\min\left\{p(x^{(i)})q(x^{(i+1)}|x^{(i)}), p(x^{(i+1)})q(x^{(i)}|x^{(i+1)})\right\} \\
\implies T(x^{(i+1)}|x^{(i)})p(x^{(i)}) &= M(x^{(i+1)},x^{(i)})
\end{align*}
$$

where $M(x^{(i+1)},x^{(i)})=\min\left\{p(x^{(i)})q(x^{(i+1)}|x^{(i)}), p(x^{(i+1)})q(x^{(i)}|x^{(i+1)})\right\}$. Likewhise, it can be shown that $T(x^{(i)}|x^{(i+1)})p(x^{(i+1)}) = M(x^{(i)},x^{(i+1)})$.

Observe that $M$ is symmetric i.e. $M(x^{(i+1)},x^{(i)})=M(x^{(i)},x^{(i+1)})$. The detailed balance condition then follows.

To ensure that the MH algo converges:

- Aperiodicity: no additional conditions since rejection is allowed.
- Irreducibility: the support of $q$ must include the support of $p$.

#### Fact 3: The normalising constant of the target distribution is not required.

#### Fact 4: Success of failure of the algo often hinges on the choice of $q$.

- If the proposal is too narrow, only one mode of $p$ might be visited.
- If the proposal is too wide, the rejection rate can be very high, resulting in high correlations.
- If all the modes are visited while the acceptance probability is high, the chain is said to "mix" well.

Below shows approximations obtained using the MH algorithm with three Gaussian proposal distributions of different variances.
![Metropolis-Hastings algorithm](mh-algo.png)

### Independent sampler

In the independent sampler, the proposal is independent of the current state, $q(x^*|x^{(i)})=q(x^*)$. Hence the acceptance probability is

$$\mathcal{A}(x^*, x^{(i)})=\min\left\{1, \frac{p(x^*)q(x^{(i)})}{p(x^{i})q(x^*)}\right\}=\min\left\{1, \frac{w(x^*)}{w(x^{(i)})}\right\}$$

This algo is close to importance sampling, but now the samples are correlated since they result from comparing one sample to the other.

### Metropolis algorithm

The Metropolis algo assumes a symmetric random walk proposal $q(x^*|x^{(i)})=q(x^{(i)}|x^*)$ e.g. isotropic Gaussian. The acceptance ratio simplifies to

$$\mathcal{A}(x^*, x^{(i)})=\min\left\{1, \frac{p(x^*)}{p(x^{i})}\right\}$$

### Gibbs sampler

Suppose we have an $n$-dimensional vector $x$ and the expressions for the full conditionals $p(x_j|x_1, ..., x_{j-1}, x_{j+1}, ..., x_n)=p(x_j|x_{-j})$. Consider the following proposal

$$
q(x^*|x^{(i)})=
\begin{cases}
p(x_j^*|x_{-j}^{(i)}) & \text{if } x_j^*=x_j^{(i)} \\
0 & \text{otherwise}
\end{cases}
$$

The acceptance probability will be

$$
\begin{align*}
\mathcal{A}(x^{(i)}, x^*)
&= \min\left\{1, \frac{p(x^*)q(x^{(i)}|x^*)}{p(x^{(i)})q(x^*|x^{(i)})}\right\} \\
&= \min\left\{1, \frac{p(x_j^*)p(x_{-j}^*)p(x_j^{(i)}|x_{-j}^*)}{p(x_j^{(i)})p(x_{-j}^{(i)})p(x_j^*|x_{-j}^{(i)})}\right\} \\
&= 1
\end{align*}
$$

---

**Gibbs sampling algo**

1. Initialise $x_{0, 1:n}$.
2. For $i=0$ to $N-1$,
   - Sample $x_1^{(i+1)}\sim p(x_1|x_2^{(i)}, x_3^{(i)}, ..., x_n^{(i)})$.
   - Sample $x_2^{(i+1)}\sim p(x_2|x_1^{(i+1)}, x_3^{(i)}, ..., x_n^{(i)})$.
   - $\vdots$
   - Sample $x_j^{(i+1)}\sim p(x_j|x_1^{(i+1)}, ..., x_{j-1}^{(i+1)}, x_{j+1}^{(i)}, ..., x_n^{(i)})$.
   - $\vdots$
   - Sample $x_n^{(i+1)}\sim p(x_n|x_1^{(i+1)}, x_2^{(i+1)}, ..., x_{n-1}^{(i+1)})$.

---

For graphical models, full conditionals reduces to conditonals on Markov blankets i.e. $p(x_j|x_{-j})=p(x_j|\text{MB}(x_j))$

### Monte carlo EM

## Reference materials

- Andrieu, C., de Freitas, N., Doucet A., Jordan, M. I. "An Introduction to MCMC for Machine Learning." _Machine Learning_, 50, 5-43, 2003. Accessed Nov 2, 2021. https://link.springer.com/content/pdf/10.1023/A:1020281327116.pdf.
- Kuleshov, V. and Ermon, S. "Sample methods." _cs228-notes_. Accessed Nov 2, 2021. https://ermongroup.github.io/cs228-notes/inference/sampling/.
- Owen, A. "Importance sampling." _Monte Carlo theory, methods and examples_. Accessed Nov 2, 2021. https://statweb.stanford.edu/~owen/mc/Ch-var-is.pdf.
- Mauser, K. "Why does the Metropolis-Hastings procedure satisfy the detailed balance criterion?" _Kris Hauser_. Accessed Nov 2, 2021. https://people.duke.edu/~kh269/teaching/notes/MetropolisExplanation.pdf.
- Bishop, C. "Sample methods." _Pattern Recognition and Machine Learning_.
