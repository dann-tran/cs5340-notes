---
title: Markov Random Fields (Undirected Graphical Models)
tag: representation
lectureNumber: 3
---

## CI properties of UGMs

### Markov properties

Global Markov property: $X_A$ is independent of $X_C$ given $X_B$ if the set of nodes $X_B$ separates the does $X_A$ from the nodes $X_C$ in the sense of naïve graph-theoretic separation. Thus, if every path from a node in $X_A$ to a node in $X_C$ includes at least one node in $X_B$, we assert that $X_A\perp X_C\vert X_B$ holds.

Additionally, if a set of observed variables form a cut-set between two halves of the graph, then variables in one half are independent from one in the other.

![A cut-set between $X_A$ and $X_C$ when $X_B$ is observed](cutset.png)

Local Markov property: A node's MB is its set of immediate neighbours.

Pairwise Markov property: Two nodes are conditionally independent given the rest if there is no direct edge between them.

It is obvious that global Markov implies local Markov which implies pairwise Markov. It can be proven that, assuming $p$ is a positive density, pairwise implies global, and hence all the Markov properties are equivalent.

### Determining CIs for a DGM using a UGM

Moralization is the process of converting a DGM to a UGM by adding edges between the unmarried parents of a node and then dropping the orientation of the edges. This is to express the correct CI from the v-structure $A\rightarrow B\leftarrow C$.

However, moralization can yield a fully connected directed graph and loses some CI information, thus we cannot use the moralized UGM to determine CI properties of the DGM. We can minimize this by first constructing the ancestral graph of DAG $G$ w.r.t. $U=A\cup B\cup C$ i.e. we remove all nodes from $G$ that are not in $U$ or are not ancestors of $U$; then we moralize this ancestral graph and apply the simple graph separation rules for UGMs.

### Comparative semantics

It is not possible to reduce undirected models to directed models or vice versa. E.g.

![(a) An undirected graph whose CI semantics cannot be captured by a directed graph on the same nodes. (b) A directed graph whose CI semantics cannot be captured by an undirected graph on the same nodes.](graph-representation-comparative-semantics.png)

DGMs and UGMs are perfect maps for different sets of distributions, so neither is more powerful than th other as a representation.

- No UGM can precisely represent all and only the two CI statements encoded by a v-structure. In general, CI properties in UGMs are monotomic i.e. if $A\perp B\vert C$ then $A\perp B\vert (C\cup D)$.
- In DGMs, CI properties can be non-monotonic, since conditioning on extra variables can eliminate CI due to explaining away.

Some distributions can be perfectly modeled by either a DGM or a UGM; the resulting graphs are called decomposable or chordal. Roughly speaking, if we collapse together all the variables in each maximal clique, to make “mega-variables”, the resulting graph will be a tree.

![Diagram of the distributions that DGMs and UGMs can represent](distribution-model-diagram.png)

## Parameterization

A clique of a graph is a fully-connected subset of nodes. The maximal cliques of a graph are the cliques that cannot be extended to include additional nodes without losing the property of being fully connected. Given that all cliques are subsets of one or more maximal cliques, we can restrict ourselves to maximal cliques WLOG, and the meaning of "local" for UDG should be "maximal clqiue." More precisely, the CI properties of UDGs imply a representation of the joint probability as a product of local functions defined on the maximal cliques of the graph.

Let $C$ be a set of indices of a maximal clique in an undirected graph $G$, and let $\mathcal C$ be the set of all such $C$. A potential function (aka factor) $\psi_C(x_C)$ is a function on the possible realizations $x_C$ of the maximal clique $X_C$. Potential functions are assumed to be nonnegative, real-valued functions, but are other artbitrary.

### Hammersley-Clifford Theorem

Hammersley-Clifford Theorem: A positive distribution $p(x)>0$ satisfies the CI properties of an undirected graph $G$ iff $p$ can be represented as a product of factors, one per maximal clique i.e.

$$
p(x)\triangleq\frac 1 Z\prod_{C\in\mathcal C}\psi_C(x_C)
$$

where $Z$ is the normalization factor

$$
Z\triangleq\sum_x\prod_{C\in\mathcal C}\psi_C(x_C)
$$

### Facts about parameterizations

Parameterization of MRFs is not unique. We a free to relax the parameterization to the edges of the graph, rather than the maximal cliques. This is pairwise MRF, and is widely used due to its simplicity, although it is not as general.

Canonical paramterization of MRFs defines the parameterization over all cliques in the graph. A uniform prior can be assumed on any potential function.

### Statistical physics interpretation

The basic idea is that a potential function favours certain local configurations of variables by assigning them a larger value. The global configurations that have a high probability are, roughly, those that satisfy as many of the vaoured local configurations as possible.

$$
\begin{align*}
\psi_C(x_C)&=\exp\{-H_C(x_C)\} \\
p(x)&=\frac 1 Z\prod_{C\in\mathcal C}\exp\{-H_C(x_C)\} \\
&=\frac 1 Z\exp\left\{-\sum_{C\in\mathcal C}H_C(x_C)\right\} \\
&=\frac 1 Z \exp\{-H(x)\}
\end{align*}
$$

where we have defined $H(x)\triangleq\sum_{C\in\mathcal C}H_C(x_C)$. We have represented the joint probability of a UDG model as a Boltzman/Gibbs distribution.

### Log-linear model

Define the log-potentials as a linear function of the parameters

$$
\log\psi_C(x_C)\triangleq \bm\phi_C(x_C)^T\bm\theta_C
$$

where $\phi_C(x_C)$ is a feature vector derived from the values of the variables $x_C$.
The resulting log-probability has the form of a maximum entroy or a log-linear momdel

$$
\log p(x)=\sum_C\bm\phi_C(x_C)^T\bm\theta_C-Z
$$

### Advantages and disadvantages of MRFs

Advantages of MRFs:

- They can be applied to a wider range of problems in which there is no natural directionality associated with variable dependencies.
- Undirected graphs can succinctly express certain dependencies that Bayesian nets cannot easily describe (although the converse is also true)

Drawbacks of MRFs:

- Computing the normalization constant $Z$ requires summing over a potentially exponential number of assignments. We will see that in the general case, this will be NP-hard; thus many undirected models will be intractable and will require approximation techniques.
- Undirected models may be difficult to interpret.
- It is much easier to generate data from a Bayesian network, which is important in some applications.

## Examples of MRFs

### Ising model

### Potts model

## Conditional random fields (CRFs)

A CRF, sometimes a discriminative random field, is a version of a MRF where all the clique potentials are conditioned on input features:

$$
p(y| x)=\frac 1 {Z(x)}\prod_C\psi_C(y_C|x_C)
$$

A CRF can be thought of as a structured output extension of logistic regression. Note that the partition function now depends on $x$, and $p(y|x)$ is a probability over $y$ and parameterized by $x$. In that sense, a CRF results in an instantiation of a new MRF for each input $x$.

In most practical applications, we further assume that the factors $\psi_C(x_C, y_C)$ are of the form

$$
\psi_C(y_C|x_C)=\exp(\mathbf w_c^T\bm\phi(x_C, y_C))
$$

where $\mathbf w_c$ are parameters.

### Advantages and disadvantages of CRFs

Advantages:

- Modelling $p(x, y)$ using an MRF (viewed as a single model over $x$, $y$ with normalizing constant $Z=\sum_{x, y}\tilde p(x, y)$) requires fitting two distributions to the data: $p(y|x)$ and $p(x)$. However, if all we are interested in is predicting $y$ given $x$, then modelling $p(x)$ is unnecessary.
- We can make the potentials (or factors) of the model data-independent.

Disadvantages: CRFs require labelled training data and are slower to train.

### Sequence labelling tasks

The most widely used kind of CRF uses a chain-structured graph to model correlation amongst neighboring labels. Such models are useful for a variety of sequence labeling tasks.

Traditionally, HMMs have been used for such tasks, but an HMM requires specifying a generatie observation model $p(\mathbf x_t|y_t, \mathbf w)$ which can be difficult. Furthermore, each $\mathbf x_t$ is required to be local, since it is hard to define a generative model for the whole stream of observations $\mathbf x_{1:T}$

$$
p(\mathbf x, \mathbf y|\mathbf w)=\prod_{t=1}^T p(y_t|y_{t-1}, \mathbf w)p(\mathbf x_t|y_t, \mathbf w)
$$

An obvious wway to make a discriminative version of an HMM is to reverse the arrows from $y_t$ to $\mathbf x=\mathbf x_t$. This defines a directed discriminative model called maximum entropy Markov model (MEMM)

$$
p(\mathbf y|\mathbf x, \mathbf w)=\prod_t p(y_t|y_{t-1}, \mathbf x, \mathbf w)
$$

where $\mathbf x=(\mathbf x_{1:T}, \mathbf x_g)$. An MEMM is simpy a Markov chain in which the state transition probabilities are conditioned on the input features.

This model suffers from the _label bias_ problem: local features at time $t$ do not influence states prior to time $t$. This follows by examining the DAG, which shows that $\mathbf x_t$ is $d$-separated from $y_{t−1}$ (and all earlier time points) by the v-structure at $y_t$, which is a hidden child, thus blocking the information flow.

E.g. POS tagging task "banks" in "he banks at BoA" and "the river banks were overflowing."

A chain-structured CRF model has the form

$$
p(\mathbf y|\mathbf x, \mathbf w)=\frac 1{Z(\mathbf x, \mathbf w)}\prod_{t=1}^T\psi(y_t|\mathbf x, \mathbf w)\prod_{t=1}^{T-1}\psi(y_t, y_{t+1}|\mathbf x, \mathbf w)
$$

and the label bias problem no longer exists because $y_t$ does not block information from $\mathbf x_t$ from reaching other $y_t$ nodes.

## Reference materials

- Jordan, M. I. (2003). Chapter 2: Conditional Independence and Factorization. In _An Introduction to Probabilistic Graphical Models_.
- Murphy, K. P. (2012). Chapter 19: Undirected graphcial models (Markov random fields). In _Machine Learning: A Probabilistic Perspective_. The MIT Press.
- Kuleshov, V. and Ermon, S. (2020). _Markov random fields_. CS228 notes. https://ermongroup.github.io/cs228-notes/representation/undirected/
