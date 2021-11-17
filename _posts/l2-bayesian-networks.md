---
title: Bayesian Networks (Directed Graphical Models)
tag: representation
lectureNumber: 2
---

## Directed graphs and joint probabilities

Consider a DAG $G=(V, E)$, we

- associate each node $i\in V$ with a r.v. $X_i$, where $x_i$ is a realisation of $X_i$,
- associate to each node $i\in V$ a local cdf $p(x_i| x_{\pi_i})$.

We define a joint probabiliy distribution as follows:

$$
p(x_1, x_2, ..., x_m)\triangleq\prod_{i=1}^mp(x_i| x_{\pi_i})
$$

## Conditional independence (CI)

$X_A$ and $X_B$ are independent, written $X_A\perp X_B$, if $p(x_A, x_B)=p(x_A)p(x_B)$.

$X_A$ and $X_C$ are conditionally independent given $X_B$, written $X_A\perp X_C\vert X_B$, if

$$
\begin{align*}
p(x_A, x_C|x_B)&=p(x_A|x_B)p(x_C|x_B) \\
\iff p(x_A|x_B, x_C)&=p(x_A|x_B)
\end{align*}
$$

for all $x_B$ s.t. $p(x_B)>0$.

Graphical models provide a symbolic approach to factoring joint probability distributions. Representing a probability distribution within the graphical model formalism involves making certain CI assumptions, assumptions which are embedded in the structure of the graph. From the graphical structure other independence relations can be derived, reflecting the fact that certain factorizations of joint probability distributions imply other factorizations. The key advantage of the graphical approach is that such factorizations can be read off from the graph via simple graph search algorithms.

Missing variables in the local CDFs correspond to missing edges in the underlying graph. Denote $\mathcal v_i$ as the set of all nodes that appear earlier than $i$ in the ordering $I$ (i.e. non-descendants), excluding the parent nodes $\pi_i$.
Given a topological ordering $I$ for a graph $G$, we associate to the graph the following set of basic CI statements:

$$
\{X_i\perp X_{\mathcal v_i}|X_{\pi_i}\}
$$

for $i\in V$. Given the parents of a node, the node is independent of all earlier nodes in the ordering. In fact, these CI conditions will imply DGM factorization from the chain rule of probability $p(x_1, ..., x_m)=\prod_{i=1}^mp(x_i|x_1, ..., x_{i-1})$.

## Three canonical three-node graphs

![Three canonical three-node graphs](canonical-three-node-graphs.png)

(a) Cascade: $X\rightarrow Y\rightarrow Z$. The missing edge in this graph corresponds to the CI statement $X\perp Z\vert Y$. Moreover, there are no other CIs associated with this graph. Interpretation: a simple Markov chain — the past is independent of the future given the present.

Proof:

$$
\begin{align*}
p(z|x, y)&=\frac{p(x, y, z)}{p(x, y)} \\
&=\frac{p(x)p(y|x)p(z|y)}{p(x)p(y|x)} \\
&=p(z|y)
\end{align*}
$$

(b) Common parent: $X\leftarrow Y\rightarrow Z$. The missing edge in this graph corresponds to the CI statement $X\perp Z\vert Y$ and no other CIs are associated with this graph. Interpretation: hidden variable $Y$ explains all of the observed dependence between $X$ and $Z$.

Proof:

$$
\begin{align*}
p(x, z|y)&=\frac{p(y)p(x|y)p(z|y)}{p(y)} \\
&=p(x|y)p(z|y)
\end{align*}
$$

(c) V-structure (aka explaining away): $X\rightarrow Y\leftarrow Z$. The missing edge in this graph corresponds to the marginal independence statement $X\perp Z$. Interpretation: observing $Y$ explains away.

Proof:

$$
\begin{align*}
p(a, b)&=\sum_cp(a, b, c) \\
&=\sum_cp(a)p(b)p(c|a, b) \\
&= p(a)p(b)
\end{align*}
$$

## Graph separation

Let $A$, $B$, $C$ be three sets of nodes in a Bayesian network $G$. We say that $A$ and $B$ are $d$-separated given $C$ if $A$ and $B$ are not connected by an active path. An undirected path in $G$ is called active given observed variables $C$ if for every consecutive triple of variables $X, Y, Z$ on the path, one of the following holds:

- head-to-tail $X\leftarrow Y\leftarrow Z$ or $X\rightarrow Y\rightarrow Z$, and $Y$ is unobserved $Y\notin C$
- tail-to-tail $X\leftarrow Y\rightarrow Z$, and $Y$ is unobserved $Y\notin C$
- head-to-head $X\rightarrow Y\leftarrow Z$, and $Y$ or any of its descendants are observed.

In the following example, $X_1$ and $X_6$ are $d$-separated given $X_2$, $X_3$.

![DGM with d-separation](dsep1.png)

However, $X_2$, $X_3$ are not $d$-separated given $X_1$, $X_6$ due to an active path passing through the V-structure created when $X_6$ is observed.

#### Markov blanket (MB)

We define the Markov blanket $U$ of a variable $X$ as the minimal set of nodes s.t. $X$ is independent of the rest of the graph if $U$ is observed.

The MB of a node $X_i$ comprises the set of its parents, children, and co-parents. Conditional distributional of $X_i$, conditioend on all remaining variables in the graph, is dependent only the variables in the MB.

#### Independence map

Denote $I(p)$ as the set of all independencies that hold for a joint distribution $p$, and $I(G)=\{(X\perp Y\vert Z)|X, Y \text{are }d\text{-sep given }Z\}$.

Fact: If $p$ factorizes over $G$, then $I(G)\subseteq I(p)$. We say that $G$ is an $I$-map for $p$.

In other words, all the independencies encoded in $G$ are sound: variables that are $d$-separated in $G$ are truly independent in $p$. However, the converse is not true: a distribution may factorize over $G$, yet have independencies that are not captured in $G$.

Question: can directed graphs express all the independencies of any distribution $p$ i.e. given a distribution $p$, can we construct a graph $G$ s.t. $I(G)=I(p)$?

First, note that it is easy to construct a $G$ s.t. $I(G)\subseteq I(p)$ e.g. a fully connected DAG which implies $I(G)=\empty$. Additionally, it is possible to find a minimal $I$-map $G$ for $p$: start with a fully connected graph and remove edges until $G$ is no longer an $I$-map following the natural topolotical ordering.

However, it is not true that any probability distribution $p$ always admits a perfect map $G$ for which $I(p)=I(G)$.

E.g. Consider distribution $p$ over three variables $X$, $Y$, $Z$ where $X, Y\sim \text{Ber}(0.5)$ and $Z=X\text{xor} Y$. We can derive that ${X\perp Y, Z\perp Y, X\perp Z}\in I(p)$ but $Z\perp{Y, X}\notin I(p)$. Thus, $X\rightarrow Z\leftarrow Y$ is an $I$-map for $p$, but non of the three-node graph structures that we discussed perfectly describes $I(p)$, and hence this distribution doesn't have a perfect map.

Perfect maps are not unique when they exist. E.g. $X\leftarrow Y$ and $X\rightarrow Y$ encode the same independencies but form different graphs.
More generally, we say that two Bayes nets $G_1$, $G_2$ are $I$-equivalent if they encode the same dependencies i.e. $I(G_1)=I(G_2)$.

Fact: If $G$, $G'$ have the same skeleton and the same v-structures, then $I(G)=I(G')$.

Reason:

- Cascade and common parent structures encode the same dependencies. Directions of the arrows can e changed as long as we don't turn them into a v-structure.
- v-structure is the only one that describes $X\not\perp Y|Z$.

## Reference materials

- Jordan, M. I. (2003). Chapter 2: Conditional Independence and Factorization. In _An Introduction to Probabilistic Graphical Models_.
- Kuleshov, V. and Ermon, S. (2021). _Bayesian networks_. CS228 notes. https://ermongroup.github.io/cs228-notes/representation/undirected/
