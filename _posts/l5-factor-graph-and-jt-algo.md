---
title: Factor Graph and Junction Tree Algorithm
tag: exact inference
lectureNumber: 5
---

## Factor graphs

The graphical model representations of UDMs and DGMs aim at characterizing probability distributions in terms of CI statements. _Factor graphs_, an alternative graphical representation of probability distributions, aim at capturing factorizations.

Given a set of variables $\{x_1, x_2, ..., x_n\}$, let $\mathcal C$ denote a set of subsets of $\{1, 2, ..., n\}$.

- $\mathcal C$ is a multiset—we allow the same subset of indices to appear multiple times. We index the members of $\mathcal C$ using an index set $\mathcal F$; thus, $\mathcal C=\{C_s:s\in\mathcal F\}$.
- To each index $s\in\mathcal F$, we associate a factor $f_s(x_{C_s})$, a function on the subset of variables index by $C_s$.
- $\mathcal C$ is an arbitrary collection of subsets of indices and do not correspond to cliques of an underlying graph.

Given a collection of subsets and the associated factors, we define a multivariate function on the variables by taking the product:

$$
f(x_1, x_2, ..., x_n)\triangleq\prod_{s=1}^Sf_s(x_{C_s})
$$

Our goal is to define a graphical representation of this function that will permit the efficient evaluation of marginal functions.

A factor graph is a bipartite graph $\mathcal G(\mathcal V, \mathcal F, \mathcal E)$, where the vertices $\mathcal V$ index the variables and the vertices $\mathcal F$ index the factors. The edges $\mathcal E$ are obtained as follows: each factor node $s\in\mathcal F$ is linked to all variable nodes in the subset $C_s$. These are the only edges in the graph.

It will prove useful to define neighbourhood functions on the nodes of a factor graph.

- Let $\mathcal N(s)\subset\mathcal V$ denote the set of neighbours of a factor node $s\in\mathcal F$.
- Let $\mathcal N(i)\subset\mathcal F$ denote the set of neighbours of a variable node $i\in\mathcal V$.

Factor graphs provide a more fine-grained representation of a probability distributions than is provided by DGMs and UGMs.

E.g. A DGM and its corresponding factor graph.
![Factor graph from a DGM](factor-graph-dgm.png)

E.g. A UGM provides no info about possible factorizations of the potential function associated with a given clique. (b) corresponds to $\psi(x_1, x_2, x_3)=f_a(x_1, x_2)f_b(x_2, x_3)f_c(x_1, x_3)$. (c) corresponds to the non-factorized potential $\psi(x_1, x_2, x_3)=f(x_1, x_2, x_3)$.

It is worth noting that it is always possible to mimic the fine-grained representation of factor graphs within the directed and undirected formalism, so that formally factor graphs provide no additional representational power.

E.g. From the factor graph above, the following undirected and directed graphs mimic the factorization.

![Undirected and directed graphs that mimic the factorization of the factor graph in the pervious diagram](ugm-dgm-factorization.png)

## The sum-product algo for factor trees

A factor graph is defined to be a factor tree if the undirected graph obtained by ignoring the distinction between variable nodes and factor nodes is an undirected tree.

---

**A sequential implementation of the sum-product algo for a factor tree $\mathcal T(\mathcal V, \mathcal F, \mathcal E)$**

$\text{Sum-Product}(\mathcal T, E)$:

1. $\text{Evidence}(E)$.
2. $f=\text{ChooseRoot}(\mathcal V)$.
3. For $s\in\mathcal N(f)$, $\mu\text{-Collect}(f, s)$.
4. For $s\in\mathcal N(f)$, $\nu\text{-Distribute}(f, s)$.
5. For $i\in\mathcal V$, $\text{ComputeMarginal}(i)$.

$\mu\text{-Collect}(i, s)$:

1. For $j\in\mathcal N(s)\setminus i$, $\nu\text{-Collect}(s, j)$.
2. $\mu\text{-SendMessage}(s, i)$

$\nu\text{-Collect}(s, i)$:

1. For $t\in\mathcal N(i)\setminus s$, $\mu\text{-Collect}(i, t)$.
2. $\nu\text{-SendMessage}(i, s)$

$\mu\text{-Distribute}(s, i)$:

1. $\mu\text{-SendMessage}(s, i)$
2. For $t\in\mathcal N(i)\setminus s$, $\nu\text{-Distribute}(i, t)$.

$\nu\text{-Distribute}(i, s)$:

1. $\nu\text{-SendMessage}(i, s)$
2. For $j\in\mathcal N(s)\setminus i$, $\mu\text{-Distribute}(s, j)$.

$\mu\text{-SendMessage}(s, i)$:

$$
\mu_{si}(x_i)=\sum_{x_{\mathcal N(s)\setminus i}}\left(f_s(x_{\mathcal N(s)})\prod_{j\in\mathcal N(s)\setminus i}\nu_{js}(x_j)\right)
$$

$\nu\text{-SendMessage}(i, s)$:

$$
\nu_{is}(x_i)=\prod_{t\in\mathcal N(i)\setminus s}\mu_{ti}(x_i)
$$

$\text{ComputeMarginal}(i)$:

$$
p(x_i)\propto v_{is}(x_i)\mu_{si}(x_i)
$$

---

The figure below shows (a) the computation of the message $v_{is}(x_i)$ that flows from the factor node $s$ to variable node $i$; (b) the computation of the message $\mu_{si}(x_i)$ that flows from variable node $i$ to factor node $s$.

![The computations of messages in a factor graph](factor-graph-message-computation.png)

Relation between sum-product for UGMs and factor graphs: $m_{ji}(x_i)$ in the undirected graph is equal to $\mu_{si}(x_i)$ in the factor graph.

$$
\begin{align*}
\mu_{si}(x_i)
&=\sum_{x_{\mathcal N(s)\setminus i}}f_s(x_{\mathcal N(s)})\prod_{j\in\mathcal N(s)\setminus i}\nu_{js}(x_j) \\
&=\sum_{x_j}\psi(x_i, x_j)\nu_{js}(x_j) \\
&=\sum_{x_j}\psi(x_i, x_j)\prod_{t\in\mathcal N(j)\setminus s}\mu_{tj}(x_j) \\
&=\sum_{x_j}\psi^E(x_j)\psi(x_i, x_j)\prod_{t\in\mathcal N'(j)\setminus s}\mu_{tj}(x_j)
\end{align*}
$$

where $\mathcal N'(j)$ denotes the neighbourhood of $j$, omitting the singleton factor node associated with $\psi^E(x_j)$.

### Tree-like graphs

In general, if the variables in a UGM can be clustered into non-overlapping cliques (tree-like graphs), and the parameterization of each clique is a general, non-factorized potential, then the corresponding factor graph is a tree, and the sum-product applies directy.

![A tree-like graph and its equivalent factor graph](treelike-graph.png)

### Polytrees

A polytree is a directed graph that reduces to an undirected tree if we convert each directed edge to an undirected edge. Thus, polytrees have no loops in their underlying undirected graph.

The factor graph corresponding to a polytree is a tree implies that the sum-product algo for factor graphs applies directly to polytrees.

![A polytree and its corresponding factor graph](polytree.png)

## MAP in factor graphs

### MAP probabilities

Given a probability distribution $p(x)$ where $x=(x_1, x_2, ..., x_n)$, given a partition $(E, F)$ of indices, and given a fixed configuration $\bar x_E$, we wish to compute the MAP probability

$$
\begin{align*}
\max_{x_F}p(x_F|\bar x_E)
&=\max_{x_F}p(x_F, \bar x_E) \\
&=\max_{x_F}p(x)\delta(x_E, \bar x_E) \\
&\triangleq\max_{x_F}p^E(x)
\end{align*}
$$

where $p^E(x)$ is the unnormalized reprsentation of conditional probability $p(x_F|\bar x_E)$.

WLOG we can study the unconditional case. I.e. we treat the general problem of maximizing a nonnegative, factorized function of $n$ variables; this includes as a special case the problem of maximizing such a function when some of the variables are held fixed.

Although the MAP problem is distinct from the marginalization problem, its algorithmic solution is quite similar.

E.g. For $p(x)=p(x_1)p(x_2|x_1)p(x_3|x_1)p(x_4|x_2)p(x_5|x_3)p(x_6|x_2, x_5)$, we can compute the MAP probability as

$$
p(x)=\max_{x_1}p(x_1)\max_{x_2}p(x_2|x_1)\max_{x_3}p(x_3|x_1)\max_{x_4}p(x_4|x_2)\max_{x_5}p(x_5|x_3)\max_{x_6}p(x_6|x_2, x_5)
$$

---

**The map-eliminate algo for solving the MAP problem**

$\text{Sum-Product}(\mathcal T, E)$:

1. $\text{Initialize}(\mathcal G)$.
2. $\text{Evidence}(E)$.
3. $\text{Update}(\mathcal G)$.
4. $\text{Maximum}$.

$\text{Initialize}(\mathcal G)$:

1. Choose an ordering $I$.
2. For each node $X_i$ in $\mathcal V$, place $p(x_i|x_{\pi_i})$ on the active list.

$\text{Update}(E)$: For each $i$ in $E$, place $\delta(x_i, \bar x_i)$ on the active list.

$\text{Update}(\mathcal G)$: For each $i$ in$I$,

1. Find all potentials from the active list that reference $x_i$ and remove them from the active list.
2. Let $\phi_i^{\text{max}}(x_{T_i})$ denote the product of these potentials.
3. Let $m_i^{\text{max}}(x_{S_i})=\max_{x_i}\phi_i^{\text{max}}(x_{T_i})$.
4. Place $m_i^{\text{max}}(x_{S_i})$ on the active list.

$\text{Maximum}$: $\max_xp^E(x)=$ the scalar value on the active list.

---

#### Underflow problem of the map-eliminate algo

The products of probabilities tend to underflow. This can be handled by transforming to the log scale:

$$
\max_xp^E(x)=\max_x \log p^E(x)
$$

We can then implement map-eliminate algo by working with logs of potentials, and replacing "product" with "sum."

#### The max-product algo on trees

In the case of trees, the eliminate algo can be equivalently expressed in terms of a coupled set of equations, or "messages," a line of argument that led to the sum-product algo for inference on trees. We ca obtain a "max-product" version of the algo as follows:

$$
\begin{align*}
m_{ji}^{\text{max}}(x_i)
&=\max_{x_j}
\left(
\psi^E(x_j)\psi(x_i, x_j)\prod_{k\in\mathcal N(j)\setminus i}m_{kj}^{\text{max}}(x_j)
\right) \\
\max_x p^E(x)&=\max_{x_i}\left(\psi^E(x_i)\prod_{j\in\mathcal N(i)}m_{ji}^{\text{max}}(x_i)\right)
\end{align*}
$$

### MAP configurations

Consider the problem of finding a configuration $x^*\in\argmax_xp^E(x)$. This problem can be solved by keepin track of the maximizing values of variables in the inward pass of the max-product algo, and using these values as indices in an outward pass.

During inward pass we maintain a record of the maximizing values $\delta_{ji}(x_i)$ of nodes when we compute the messages $m_{ji}^{\text{max}}(x_i)$. We then use $\delta_{ji}(x_i)$ to define a consistent maximizing configuration during an outward pass. Starting at the root $f$, we choose a maximizing value $x_f^*$. Given this value, which we pass to the children of $f$, we set $x_e^*=\delta_{ef}(x_f^*)$ for each $e\in\mathcal N(f)$. This procedure continues outward to the leaves.

---

**A sequential implementation of the max-product algo for a tree $\mathcal T(\mathcal V, \mathcal E)$**

$\text{Max-Product}(\mathcal T, E)$:

1. $\text{Evidence}(E)$
1. $f=\text{ChooseRoot}(\mathcal V)$
1. For $e\in\mathcal N(f)$: $\text{Collect}(f, e)$.
1. $\text{MAP}=\max_{x_f}\left(\psi^E(x_f)\prod_{e\in\mathcal N(f)}m_{ef}^{\text{max}}(x_f)\right)$
1. $x_f^*=\argmax_{x_f}\left(\psi^E(x_f)\prod_{e\in\mathcal N(f)}m_{ef}^{\text{max}}(x_f)\right)$
1. For $e\in\mathcal N(f)$: $\text{Distribute}(f, e)$.

$\text{Collect}(i, j)$:

1. For $k\in \mathcal N(j)\setminus i$: $\text{Collect}(j, k)$.
1. $\text{SendMessage}(j, i)$

$\text{Distribute}(i, j)$:

1. $\text{SetValue}(i, j)$
1. For $k\in \mathcal N(j)\setminus i$: $\text{Distribute}(j, k)$.

$\text{SendMessage}(j, i)$:

$$
m_{ji}^{\text{max}}(x_i)=\max_{x_j}\left(\psi^E(x_j)\psi(x_i, x_j)\prod_{k\in\mathcal N(j)\setminus i}m_{kj}^{\text{max}}(x_j)\right)
$$

$$
\delta_{ji}(x_i)\in\argmax_{x_j}\left(\psi^E(x_j)\psi(x_i, x_j)\prod_{k\in\mathcal N(j)\setminus i}m_{kj}^{\text{max}}(x_j)\right)
$$

$\text{SetValue}(i, j)$: $x_j^*=\delta_{ji}(x_i^*)$.

---

## The junction tree algo

### Junction trees

The elimination algo is "query-oriented" and discards intermediated factors that are created along the way, thus requiring a restart for every new query. We wish to avoid recomputing such factors.

The junction tree algo partitions the graph into clusteres of variables, and interactions among clusters will have a tree structure. this leads to tractable global solutions if the local (cluster-level) problems can be solved exactly.

Suppose we have an UGM $G$ (if the model is directed, we consider its moralized graph). A junction tree $T=(\mathcal C, E_T)$ over $G=(\mathcal X, E_G)$ is a tree whose nodes $c\in C$ are associated with subsets $x_C\subseteq\mathcal X$ of the graph vertices (i.e. sets of variables); the junction tree must satisfy the following properties:

- Family preservation: For each $\phi$, there is a cluster $c$ s.t. $\text{Scope}[\phi]\subseteq x_C$.
- Running intersection: For every pair of clusters $C^{(i)}$, $C^{(j)}$, every cluster on the path between $C^{(i)}$, $C^{(j)}$ contains $x_C^{(i)}\cap x_C^{(j)}$.

Below shows an MRF with graph $G$ and junction tree $T$. MRF potentials are dentoed using different colours; circles indicate nodes of the junction trees; rectangular nodes represent sepsets (short for "separation sets"), which are sets of variables shared by neighbouring clusters.

![An MRF and its junction tree](junction-tree.png)

Note that we may always find a trivial junction tree with one node containing all the variables in the original graph. However, such trees are useless because they will not result in efficient marginalization algos.

Optimal trees are ones that make the clusters as small and modular as possible; unfortunately, it is again NP-hard to find the optimal tree. A special case when we can find the optimal trees is when $G$ itself is a tree. In that case, we may define a cluster for each edge in the tree.

### The junction tree algo

At a high level, the algo implements a form of message passing on the junction tree, which is equivalent to variable elimination for the same reasons that belief propagation is equivalent to variable elimination.

Define the potential $\psi_C(x_C)$ of each cluster $c$ as the product of all the factors $\phi$ in $G$ that have been assigned to $C$. By the family preservation property, this is well-defined, and we may assume that our distribution is in the form

$$
p(x_1, ..., x_n)=\frac 1 Z\prod_{C\in\mathcal C}\psi_C(x_C)
$$

At each step of the algo, we choose a pair of adjacent clusters $C^{(i)}$, $C^{(j)}$ in $T$ and compute a message whose scope is the sepset $S_{ij}$ between the two clusters:

$$
m_{ij}(S_{ij})=\sum_{x_c\setminus S_{ij}}\psi_C(x_C)\prod_{\ell\in\mathcal N(i)\setminus j}m_{li}(S_{li})
$$

We choose $C^{(i)}$, $C^{(j)}$ only if $C^{(i)}$ has received messages from all of its neighbours except $C^{(j)}$. Just as in belief propagation, this procedure will terminate in exactly $2\vert E_T\vert$ steps. After it terminates, we will define the belief of each cluster based on all the messages that it receives

$$
\beta_C(x_C)=\psi_C(x_C)\prod_{\ell\in\mathcal N(i)}m_{\ell i}(S_{\ell i})
$$

These updates are often referred to as Shafer-Shenoyy. After all the messages have been passed, beliefs will be proportional to the marginal probabilities over their scopes i.e. $\beta_C(x_C)\propto p(x_C)$. We may answer queries of the form $\tilde p(x)$ for $x\in x_C$ by marginalizing out the variable in its belief

$$
\tilde p(x)=\sum_{x_C\setminus x}\beta_C(x_C)
$$

To get the actual normalized probability, we take $p(x)=\frac 1 Z \tilde p(x)$ where $Z=\sum_{x_C}\beta_C(x_C)$

The running time is exponential in the size of the largest cluster, because we may need to marginalize out variables from the cluster, which often must be done using brute force.

## Loopy belief propagation (LBP)

Unlike the junction tree algo which atempted to efficiently find the exact solution, LBP will form our first example of an approxmiate inference algo.

Suppose that we are given an MRF with pairwise potentials. The main idea of LBP is to disregard loops in the graph and perform message passing anyway. In other words, given an ordering on the edges, at each time $t$ we iterate over a pair of adjacent variables $x_i$, $x_j$ in that order and simply perform the update

$$
m_{ij}^{t+1}(x_j)=\sum_{x_i}\psi(x_i)\psi(x_i, x_j)\prod_{\ell\in\mathcal N(i)\setminus j}m_{\ell i}^t(x_i)
$$

We keep performing these updates for a fixed number of steps or until convergence (the messages don't change). Messages are typically initialized uniformly.

This heuristic approach often works surprisingly well in practice. In general, however, it may not converge. We know for example that it probably converges on trees and on graphs with at most one cycle. If the method does converge, its beliefs may not necessarily equal the true marginals, although very often in practice they will be close.
