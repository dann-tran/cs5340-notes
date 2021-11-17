---
title: Variable Elimination and Belief Propogation
tag: exact inference
lectureNumber: 4
---

## Probabilistic inference

Probabilistic inference problem: Let $E$ and $F$ be disjoint subsets of the node indices of a graphical model, s.t. $X_E$ (evidence nodes) and $X_F$ (query nodes) are disjoint subsets of r.v. in the domain. Our goal is to calculate $p(x_F|x_E)$.

Suppose that $V=E\cup F\cup R$, then we want to compute

$$p(x_F|x_E)=\frac{p(x_E, x_F)}{p(x_E)}=\frac{\sum_{x_R}p(x)}{\sum_{x_F}p(x_E, x_F)}$$

A naïve summation over the joint distribution of $m$ variables that take $k$ states will incur a computational complexity of $O(k^n)$. To reduce computational complexity, we can exploit the factorization of the joint probability.

Let $X_i$ be an evidence node whose observed value is $\bar x_i$. Define an evidence potential $\delta(x_i, \bar x_i)=\mathbb I(x_i=\bar x_i)$. Then $g(\bar x_i)=\sum_{x_i}g(x_i)\delta(x_i, \bar x_i)$.

Total evidence potential: $\delta(x_E, \bar x_E)\triangleq\prod_{i\in E}\delta(x_i, \bar x_i)$. Thus

$$
\begin{align*}
p(x_F, \bar x_E)&=\sum_{x_E}p(x_F, x_E)\delta(x_E, \bar x_E) \\
p(\bar x_E)&=\sum_{x_F}\sum_{x_E}p(x_F, x_E)\delta(x_E, \bar x_E)
\end{align*}
$$

This suggests that it may be useful to define a generalized measure that represents conditional probability w.r.t. $E$:

$$
p^E(x)\triangleq p(x)\delta(x_E, \bar x_E)
$$

By formally "marginalizing" this measure w.r.t. $x_E$, we evaluate $p(x)$ at $X_E=\bar x_E$, and obtain $p(x_F, \bar x_E)$, an unnormalized version of the conditional probability $p(x_F|\bar x_E)$. I.e. $p(x_F, \bar x_E)=\sum_{x_E}p^E(x)$.

This tactic is particularly natural in the case of of UDMs, where multiplication by an evidence potential $\delta(x_i, \bar x_i)$ can be implemented by simiply redefining the local potentials $\psi(x_i)$ for $i\in E$. Thus we define $\psi_i^E(x_i)\triangleq\psi_i(x_i)\delta(x_i, \bar x_i)$ for $i\in E$. Leaving all other clique potentials unchanged i.e. $\psi_C^E(x_C)\triangleq\psi_C(x_C)$ for $C\notin\{\{i\}:i\in E\}$, we obtain the desired unnormalized representation:

$$
p^E(x)\triangleq\frac 1 Z\prod_{C\in\mathcal C}\psi_C^E(x_C)
$$

### Elimination and directed graphs

---

**ELIMINATE algo for probabilistic inference on directed graphs**

$\text{Eliminate}(G, E, F)$:

1. $\text{Initialize}(G, F)$
2. $\text{Evidence}(E)$
3. $\text{Update}(G)$
4. $\text{Normalize}(F)$

$\text{Initialize}(G, F)$:

1. Choose an ordering $I$ s.t. $F$ appears last.
2. for each node $X_i$ in V, place $p(x_i|x_{\pi_i})$ on the active list.

$\text{Evidence}(E)$: for each $i$ in $E$, place $\delta(x_i, \bar x_i)$ on the active list.

$\text{Update}(G)$: for each $i$ in $I$

1. Find all potentials from the active list that reference $x_i$ and remove them from the active list.
2. Let $\phi_i(x_{T_i})$ denote the product of these potentials.
3. Let $m_i(x_{S_i})=\sum_{x_i}\phi_i(x_{T_i})$.
4. Place $m_i(x_{S_i})$ on the active list.

$\text{Normalize}(F)$: $p(x_F|\bar x_E)\leftarrow \phi_F(x_F)/\sum_{x_F}\phi_F(x_F)$

---

Example:
![A DGM](dgm.png)

$$
\begin{align*}
p(x_1, \bar x_6)&=\sum_{x_2}\sum_{x_3}\sum_{x_4}\sum_{x_5}p(x_1)p(x_2|x_1)p(x_3|x_1)p(x_4|x_2)p(x_5|x_3)p(\bar x_6|x_2, x_5) \\
&=p(x_1)\sum_{x_2}p(x_2|x_1)\sum_{x_3}p(x_3|x_1)\sum_{x_4}p(x_4|x_2)\sum_{x_5}p(x_5|x_3)p(\bar x_6|x_2, x_5) \\
&=p(x_1)\sum_{x_2}p(x_2|x_1)\sum_{x_3}p(x_3|x_1)\sum_{x_4}p(x_4|x_2)m_5(x_2, x_3) \\
&=p(x_1)\sum_{x_2}p(x_2|x_1)\sum_{x_3}p(x_3|x_1)m_5(x_2, x_3)\sum_{x_4}p(x_4|x_2) \\
&=p(x_1)\sum_{x_2}p(x_2|x_1)m_4(x_2)\sum_{x_3}p(x_3|x_1)m_5(x_2, x_3) \\
&=p(x_1)\sum_{x_2}p(x_2|x_1)m_4(x_2)m_3(x_1, x_2) \\
&=p(x_1)m_2(x_1)
\end{align*}
$$

where $m_i(x_j)=\sum_{x_i}f(x_j)$. We can then calculate the inference:

$$
p(x_1|\bar x_6)\frac{p(x_1)m_2(x_1)}{\sum_{x_1}p(x_1)m_2(x_1)}
$$

### Elimination and undirected graphs

The same perspective of DGMs applies to UGMs, and the entire $\text{Eliminate}$ algo goes through without essential change to the undireted case, except for the $\text{Initialize}$ procedure where instead of using local probabilites we initialzie the active list to contain the potentials $\{\psi_C(x_C)\}$.

When calculating conditonal probabilities, the normalization factor $Z$ cancels:

$$
p(x_F|x_E)=\frac{\frac 1 Z m_{E, R}(x_F)}{\sum_{x_F}\frac 1 Z m_{E, R}(x_F)}=\frac{ m_{E, R}(x_F)}{\sum_{x_F}m_{E, R}(x_F)}
$$

But for a marginal probability $Z$ does not cancel and must be calculated explicitly.

## Graph elimination

---

**A simple greedy algo for eliminating nodes in an undirected graph $G$**

$\text{UndirectedGraphEliminate}(G, I)$: for each node $X_i$ in $I$

1. Connect all of the remaining neighbours of $X_i$.
2. Remove $X_i$ from the graph.

---

Reconstuted graph: the graph $\tilde G=(V, \tilde E)$, whose edge set $\tilde E$ is a superset of $E$, incorporating all of the original edges $E$, as well as any new edges created during a run of $\text{UndirectedGraphEliminate}$.

The elimination process adds new edges between (remaining) neighbours of the node. This creates new elimination cliques in the graph, and the overall complexity depends on the size of the largest elimination clique, which depends on the choice of elimination ordering.

---

**An algo for eliminating nodes in a directed graph $G$**

$\text{DirectedGraphEliminate}(G, I)$:

1. $G^m=\text{Moralize}(G)$
2. $\text{UndirectedGraphEliminate}(G^m, I)$

$\text{Moralize}(G)$:

1. For each node $X_i$ in $I$, connect all of the parents of $X_i$.
2. Drop the orientation of all edges.
3. Return $G$.

---

To define the elimination cliques for a directed graph, run $\text{DirectedGraphEliminate}$.

## Probabilistic inference on trees

Definitions:

- In the undirected case: a tree is an undirected graph in which there is one and only one path between any pair of nodes.
- In the directed case, a tree is any graph whose moralized graph is an undirected tree.

Any undirected tree can be converted into a directed tree by choosing a root node and orienting all edges to point away from the root. From the POV of graphcial model representations, a directed tree and the corresponding undirected tree make exactly the same set of CI assertions.

### Parameterization and conditioning

Let us consider the parameterization of probability distributions on undirected trees. The cliques are single nodes and pairs of nodes, and thus the joint probability can be parameterized via potential functions $\{\psi(x_i)\}$ and $\{\psi(x_i, x_j)\}$. In particular, for a tree $T(V, E)$ we have:

$$
p(x)=\frac 1 Z\left(\prod_{i\in V}\psi(x_i)\prod_{(i, j)\in E}\psi(x_i, x_j)\right)
$$

For a directed tree, the joint probability is formed by taking a product over a marginal probability $p(x_r)$ at the root node $r$, and the conditional probabilities $\{p(x_j|x_i)\}$ at all other nodes:

$$
p(x)=p(x_r)\prod_{(i, j)\in E}p(x_j|x_i)
$$

where $(i, j)$ is a directed edge s.t. $i$ is the (uniqe) parent of $j$ i.e. $\{i\}=\pi_j$.

We treat the parameterization for a directed tree a special case of that for an undirected tree i.e.

- $\psi(x_r)=p(x_r)$
- $\psi(x_i, x_j)=p(x_j|x_i)$ for $i$ is the parent of $j$
- $\psi(x_i)=1$ for $i\neq r$

We use evidence potentials to capture conditioning. Specifically,

$$
\psi_i^E(x_i)\triangleq
\begin{cases}
\psi_i(x_i)\delta(x_i, \bar x_i) &i\in E \\
\psi_i(x_i) & i\notin E
\end{cases}
$$

$$
p(x|\bar x_E) = \frac 1{Z^E}\left(\prod_{i\in V}\psi^E(x_i)\prod_{(i, j)\in E}\psi(x_i, x_j)\right)
$$

The parameterization of unconditional distributions and conditional distributions on trees is formally identical, involving a product of potential functions associated with each node and each edge in the graph. We can thus proceed without making any special distinction between the unconditional case and the conditional case.

### Message passing

The basic structure of $\text{Eliminate}$:

- (1) Choose an elimination ordering $I$ in which the query node $f$ is the final node.
- (2) Place all potentials on an active list.
- (3) Eliminate a node $i$ by removing all potentials referencing the node from the active list, taking the product, summing over $x_i$ and placing the resulting intermediate factor back on the list.

(1) To take the advantage of the recursive sturcture of a tree, we specify an elimination ordering $I$ that arises from a depth-first traversal of the tree and in which a node is eliminated only after all of its children in the directed version of the tree are eliminated. It can be easily everified that such an eliminiation ordering proceeds inward from the leaves, and generates elimination cliques of size at most two. This implies that the tree-width of a tree is equal to one.

(3) Consider neighbouring nodes $i$ and $j$ where $i$ is closer to the root than $j$. We are interested in the intermediate factor careted when $j$ is eliminated. We can show that the intermediate factor created by the sum over $x_j$ is a function solely of $x_i$, denoted as $m_{ji}(x_i)$. This is because none of the potentials in the product can reference any variable in the subtree below $j$. We then just need to consider the potentials that reference $x_j$:

- $\psi^E(x_j)$
- $\psi(x_i, x_j)$
- $\prod_{k\in\mathcal N(j)\setminus i}m_{kj}(x_j)$: messages folowing towards $j$

### The sum-product algo

We can obtain all marginals by simply doubling the amount of work required to compute a single marginal by passing messages inward from leaves to root then from root to leaves again. A single message will flow in both directions along each edge.

Message-Passing Protocol: A node can send a message to a neighbouring node when (and only when) it has received messages from all of its other nieghbours.

---

**The $\text{Sum-Product}$ algo**

$\text{Sum-Product}(T, E)$:

1. $\text{Evidence}(E)$
1. $f=\text{ChooseRoot}(V)$
1. For $e\in\mathcal N(f)$: $\text{Collect}(f, e)$.
1. For $e\in\mathcal N(f)$: $\text{Distribute}(f, e)$.
1. For $i\in\mathcal V$: $\text{ComputeMarginal}(i)$.

$\text{Evidence}(E)$:

1. For $i\in E$: $\psi^E(x_i)=\psi(x_i)\delta(x_i, \bar x_i)$.
1. For $i\notin E$: $\psi^E(x_i)=\psi(x_i)$.

$\text{Collect}(i, j)$:

1. For $k\in \mathcal N(j)\setminus i$: $\text{Collect}(j, k)$.
1. $\text{SendMessage}(j, i)$

$\text{Distribute}(i, j)$:

1. $\text{SendMessage}(i, j)$
1. For $k\in \mathcal N(j)\setminus i$: $\text{Distribute}(j, k)$.

$\text{SendMessage}(j, i)$:

$$
m_{ji}(x_i)=\sum_{x_j}\left(\psi^E(x_j)\psi(x_i, x_j)\prod_{k\in\mathcal N(j)\setminus i}m_{kj}(x_j)\right)
$$

$\text{ComputeMarginal}(i)$

$$
p(x_i)\propto\psi^E(x_i)\prod_{j\in\mathcal N(i)}m_{ji}(x_i)
$$

---

![Diagram for sum-product algo](sum-product.png)

## Reference materials

- Jordan, M. I. (2003). Chapter 3: The Elimination Algorithm. In _An Introduction to Probabilistic Graphical Models_.
- Jordan, M. I. (2003). Chapter 4: Probability Propagation and Factor Graphs. In _An Introduction to Probabilistic Graphical Models_.
