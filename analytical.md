# Analytical solution of the generalized two-player game

Normalize rewards by setting

[
p:=\frac{P}{R},
\qquad
\widehat u_i:=\frac{u_i}{R}.
]

The players choose failure probabilities (a,b\in[0,1]). With Gaussian-copula joint failure probability

[
C_\rho(a,b)
===========

\Phi_2!\left(\Phi^{-1}(a),\Phi^{-1}(b);\rho\right)
]

and logistic market friction

[
s_\tau(a-b)=\frac{1}{1+e^{-(a-b)/\tau}},
]

the normalized payoff is

[
\boxed{
U(a,b)
======

b-C_\rho(a,b)-pa
+
\left[1-a-b+C_\rho(a,b)\right]s_\tau(a-b).
}
\tag{1}
]

This is exactly Proposition 4.2 and the implementation used by the repository. ([arXiv][1])

A useful identity, valid pointwise for every (\rho,\tau,a,b), is

[
\boxed{
U(a,b)+U(b,a)
=============

1-C_\rho(a,b)-p(a+b).
}
\tag{2}
]

The friction parameter cancels from total utility at fixed actions.

The equilibrium changes type discontinuously when (\tau) becomes positive:

[
\begin{array}{c|c}
\tau=0 & \text{continuous density on an interval}\
\tau>0,\ \rho=0 & \text{finite atomic distribution}
\end{array}
]

That distinction is essential. A positive-(\tau) equilibrium generally does **not** have a density.

---

## 1. Correlated failures with no market friction: (\rho\neq0,\ \tau=0)

When (\tau=0), ignoring the zero-probability tie event,

[
U(a,b)=
\begin{cases}
1-(p+1)a,&a>b,[2mm]
b-C_\rho(a,b)-pa,&a<b.
\end{cases}
\tag{3}
]

### 1.1 Form of the equilibrium

For (p>0), the equilibrium has no atoms and its support is an interval

[
[0,h].
]

The argument is direct:

* A positive lower endpoint cannot occur, because lowering one’s action while remaining below every opponent action decreases both risk and penalty.
* A gap cannot occur, because throughout a gap the player’s ranking against every opponent action is unchanged while the payoff strictly decreases with their own risk.
* An atom at (x) can be overcut by (x+\varepsilon). The limiting gain on the atom is

[
\frac12\left[1-2x+C_\rho(x,x)\right],
]

the probability that both players survive divided by two, whereas the additional risk cost is (O(\varepsilon)).

Thus write the equilibrium density as (f), its CDF as (F), and its mean as

[
\mu=\int_0^h b f(b),db.
]

The expected payoff from pure action (a\in[0,h]) is

[
V(a)
====

\left[1-(p+1)a\right]F(a)
+
\int_a^h
\left[b-C_\rho(a,b)-pa\right]f(b),db.
]

Since (V(0)=\mu), equilibrium indifference (V(a)=V(0)) gives

[
\boxed{
pa+\int_a^h C_\rho(a,b)f(b),db
==============================

\int_0^a(1-a-b)f(b),db.
}
\tag{4}
]

This is equivalent to the integral identity used in Appendix C of the paper, but it applies to every (\rho), not only (\rho=\pm1). ([arXiv][1])

At (a=h), equation (4) gives the useful endpoint identity

[
\boxed{
\mu=1-(p+1)h.
}
\tag{5}
]

---

### 1.2 Exact Fredholm-resolvent solution

Define

[
D_\rho(a):=1-2a+C_\rho(a,a),
\tag{6}
]

which is the probability that both players survive when both choose (a), and

[
q_\rho(a,b)
:=
\frac{\partial C_\rho(a,b)}{\partial a}.
]

For (-1<\rho<1),

[
\boxed{
q_\rho(a,b)
===========

\Phi!\left(
\frac{\Phi^{-1}(b)-\rho\Phi^{-1}(a)}
{\sqrt{1-\rho^2}}
\right).
}
\tag{7}
]

Differentiating (4) yields

[
\boxed{
D_\rho(a)f(a)
=============

p+F(a)+
\int_a^h q_\rho(a,b)f(b),db.
}
\tag{8}
]

Define the positive integral operator on ([0,h])

[
(K_{\rho,h}g)(a)
================

\frac{
\displaystyle
\int_0^a g(b),db+
\int_a^h q_\rho(a,b)g(b),db
}{
D_\rho(a)
},
\tag{9}
]

and put

[
d_\rho(a)=\frac1{D_\rho(a)}.
]

Equation (8) is

[
f=p,d_\rho+K_{\rho,h}f.
]

Therefore the exact density is

[
\boxed{
f_{\rho,p,h}
============

# p,(I-K_{\rho,h})^{-1}d_\rho

p\sum_{n=0}^{\infty}K_{\rho,h}^{,n}d_\rho.
}
\tag{10}
]

This is a convergent analytical series, not a finite-grid approximation. Each term is an explicitly known iterated integral involving only the normal CDF.

The support endpoint is the unique solution of the scalar equation

[
\boxed{
p\int_0^h
\left[(I-K_{\rho,h})^{-1}d_\rho\right](a),da
=1.
}
\tag{11}
]

The uniqueness follows from positivity: as (h) increases, (K_{\rho,h}) increases in positive-operator order, so the mass in (11) increases strictly from zero until the first Fredholm singularity.

For (p=0), the inhomogeneous term disappears. The exact solution is instead the normalized Perron eigenfunction:

[
\boxed{
K_{\rho,h}f=f,\qquad
r(K_{\rho,h})=1,\qquad
\int_0^h f(a),da=1.
}
\tag{12}
]

Equations (10)–(12) solve the correlation-only game for arbitrary Gaussian-copula correlation.

### 1.2b Backward shooting formulation (used by `analytical.py`)

Since (f=F') and (F(h)=1), integration by parts of the last term of (8) gives

[
\int_a^h q_\rho(a,b)f(b),db
=
q_\rho(a,h)-q_\rho(a,a)F(a)-\int_a^h \partial_b q_\rho(a,b)F(b),db,
]

so (8) becomes

[
\boxed{
D_\rho(a)F'(a)
=
p+q_\rho(a,h)+\left[1-q_\rho(a,a)\right]F(a)
-\int_a^h \partial_b q_\rho(a,b)F(b),db.
}
\tag{12a}
]

The right side at (a) only uses (F) on ([a,h]). Thus (12a) is a Volterra equation in the backward direction: start at (F(h)=1) and march down to (a=0). The endpoint is the root of the scalar equation

[
\boxed{
R(h):=F_h(0)=0,
\qquad 0<h<\frac1{p+1},
}
\tag{12b}
]

where the upper bound comes from (5) with (\mu>0). This replaces the mass condition (11) and the Perron condition (12), for (p>0) and for (p=0), with no linear system.

Numerical scheme:

* Grid (a_i=h,(i/N)^2). The finer spacing near (0) is necessary: near (a=0), (q_\rho(a,a)\sim a^{(1-\rho)/(1+\rho)}), so (f) has a power-law term at (0). On a uniform grid the error in (h) is only (O(N^{-1.2})) for (\rho=0.5); on the graded grid it is (O(N^{-2})) for every (\rho) tested.
* The integral in (12a) uses (\sum_j [q_\rho(a_i,a_{j+1})-q_\rho(a_i,a_j)],\tfrac12(F_j+F_{j+1})). This needs only (q_\rho), not (\partial_b q_\rho), and gives the exact mass of the kernel on each cell.
* Each step is a Heun predictor-corrector step. Each step is (O(N)), so one evaluation of (R(h)) is (O(N^2)), and Brent's method finds the root of (12b).
* (D_\rho(a)) is evaluated as (C_\rho(1-a,1-a)) (radial symmetry of the Gaussian copula). This keeps full relative precision near (a=1), where (1-2a+C_\rho(a,a)) cancels to zero.

Accuracy with (N=2000) (reference: Richardson extrapolation from (N=1000,2000,4000)):

| (p) | (\rho) | (h) | error, resolvent (11) on a uniform grid of 1200 | error, shooting (12b) |
|---|---|---|---|---|
| 0 | 0.5 | 0.607578496 | −6.3e−6 | −2.5e−8 |
| 1 | 0.5 | 0.386868970 | −6.3e−6 | −8.5e−9 |
| 1 | 0.9 | 0.391725685 | −5.9e−8 | −2.8e−9 |
| 1 | −0.5 | 0.377578302 | +9.6e−9 | −1.4e−8 |
| 1 | −0.9 | 0.375079385 | −1.0e−7 | −4.2e−8 |
| 5 | 0.5 | 0.153081283 | −2.4e−6 | −1.4e−9 |

---

### 1.3 Closed-form checks

The operator solution reduces to elementary expressions at the three values for which simplification is possible.

#### Independent failures: (\rho=0)

Let

[
k=\sqrt{(p+1)^2+1}.
]

Then

[
\boxed{
f(a)=\frac{k-1}{(1-a)^3},
\qquad
0\le a\le
h=\frac{p+2-k}{p+1},
}
\tag{13}
]

and

[
\boxed{
\mu=k-p-1.
}
\tag{14}
]

This is the paper’s original two-player equilibrium. ([arXiv][1])

#### Perfect positive correlation: (\rho=1)

[
\boxed{
f(a)=\frac{p+1}{1-a},
\qquad
h=1-e^{-1/(p+1)},
}
\tag{15}
]

and

[
\boxed{
\mu=(p+1)e^{-1/(p+1)}-p.
}
\tag{16}
]

#### Perfect negative correlation: (\rho=-1)

For (p>0),

[
\boxed{
f(a)=p(1-2a)^{-3/2},
\qquad
h=\frac{2p+1}{2(p+1)^2},
}
\tag{17}
]

and

[
\boxed{
\mu=\frac1{2(p+1)}.
}
\tag{18}
]

Equations (15)–(18) agree with Theorem 4.3 of the paper. ([arXiv][1])

---

### 1.4 Exact utility relation for arbitrary correlation

At (\tau=0), choosing zero risk against an atomless equilibrium gives payoff equal to the opponent’s mean risk. Hence each player’s normalized equilibrium utility is

[
v=\mu.
]

Restoring (R),

[
\boxed{
u_1^*=u_2^*=R\mu,
\qquad
u_{\mathrm{total}}^*=2R\mu.
}
\tag{19}
]

Thus when (\rho) changes while (\tau=0), the utility-risk relationship is not merely approximately linear:

[
\boxed{
u_{\mathrm{total}}^*=2R\bar r.
}
\tag{20}
]

---

## 2. Independent failures with positive friction: (\rho=0,\ \tau>0)

Now

[
\boxed{
U(a,b)
======

b(1-a)-pa+
(1-a)(1-b)s_\tau(a-b).
}
\tag{21}
]

The nature of the solution changes completely.

---

### 2.1 The equilibrium has finite support

For any opponent distribution (G), define

[
V_G(a)=\int U(a,b),dG(b).
]

As a complex function of (a), (V_G) is holomorphic in the strip

[
|\operatorname{Im}a|<\pi\tau,
]

because the nearest poles of the logistic function are at imaginary distance (\pi\tau).

Suppose an equilibrium strategy had infinite support. Compactness of ([0,1]) would give an accumulation point. Since every support point satisfies (V_G(a)=v), the identity theorem would imply

[
V_G(a)\equiv v
]

throughout the strip.

But along the positive real axis,

[
V_G(a)=1-(p+1)a+o(1)
\qquad(a\to+\infty),
]

which is not constant. This is a contradiction.

Therefore:

[
\boxed{
\text{For every }\tau>0\text{ and }\rho=0,\text{ every Nash equilibrium has finite support.}
}
\tag{22}
]

The broad distributions produced by regret matching are grid-smeared approximations to finite collections of atoms.

---

### 2.2 Exact finite system for all equilibria

Consider a symmetric equilibrium with support

[
0\le x_1<\cdots<x_m\le1
]

and probabilities (w_1,\ldots,w_m>0). Let

[
A_{ij}=U(x_i,x_j),
\qquad
B_{ij}=\frac{\partial U}{\partial a}(x_i,x_j).
]

For fixed nodes, the probabilities and equilibrium value are determined by

[
\boxed{
\begin{pmatrix}
A&-\mathbf 1\
\mathbf 1^\top&0
\end{pmatrix}
\begin{pmatrix}
w\v
\end{pmatrix}
=============

\begin{pmatrix}
0\1
\end{pmatrix}.
}
\tag{23}
]

Every interior support point also satisfies

[
\boxed{
(Bw)_i=0.
}
\tag{24}
]

Finally,

[
\boxed{
\sum_jw_jU(a,x_j)\le v
\qquad\text{for every }a\in[0,1].
}
\tag{25}
]

Equations (23)–(25) are necessary and sufficient. Because the support is finite, they constitute a finite analytical solution of the game. Increasing (m) must terminate at the equilibrium support.

For asymmetric equilibria, use separate node and probability vectors for the two players and impose the corresponding payoff, tangency, normalization, and global-inequality conditions for each player.

---

## 3. Exact frictional phase diagram

There are three types of frictional equilibrium:

1. zero-risk pure equilibrium;
2. positive-risk pure equilibrium;
3. finite atomic mixed equilibrium.

---

### 3.1 Zero risk is strictly dominant above an exact threshold

Define

[
\boxed{
\tau_0(p)=\frac1{2(2p+1)}.
}
\tag{26}
]

Restoring (R),

[
\boxed{
\tau_0(P,R)
===========

\frac{R}{2(2P+R)}.
}
\tag{27}
]

For every copula—not only independent failures—

[
\boxed{
\tau\ge\tau_0
\quad\Longleftrightarrow\quad
a=0\text{ is the unique equilibrium action.}
}
\tag{28}
]

A proof follows from two observations.

First, (U) is decreasing in the joint-failure probability (C), so its largest possible value is obtained from the Fréchet lower bound

[
C\ge\max(a+b-1,0).
]

When (a+b\ge1),

[
U(a,b)\le1-(p+1)a\le b\le U(0,b).
]

When (a+b\le1), define

[
H(a)=b-pa+(1-a-b)s_\tau(a-b).
]

Using (4s(1-s)\le1), one obtains, for (\tau\ge\tau_0),

[
H'(a)
\le
\frac12-s_\tau(a-b)
-\left(p+\frac12\right)(a+b)
\le0.
]

Thus (U(0,b)>U(a,b)) for every (a>0).

Conversely,

[
\left.\frac{\partial U(a,0)}{\partial a}\right|_{a=0}
=====================================================

-p-\frac12+\frac1{4\tau}.
]

This is positive when (\tau<\tau_0), so zero cannot be an equilibrium below the threshold.

---

### 3.2 Positive pure equilibrium for (\rho=0)

At a symmetric positive pure equilibrium (a=b=r), the first-order condition is

[
(1-r)^2=2\tau(2p+1+r).
\tag{29}
]

The relevant root is

[
\boxed{
r_p(\tau)
=========

1+\tau-\sqrt{\tau^2+4\tau(p+1)}.
}
\tag{30}
]

This is positive precisely when (\tau<\tau_0(p)).

A stationary point is not automatically an equilibrium. Against opponent action (r), the only possible global competitors are (a=r) and (a=0). This follows from

[
\frac{\partial^2U(a,r)}{\partial a^2}
=====================================

\frac{(1-r)s(1-s)}{\tau^2}
\left[(1-a)(1-2s)-2\tau\right],
\tag{31}
]

whose bracket is negative for (a\ge r) and crosses zero at most once for (a<r).

Therefore (r) is a best response to itself exactly when

[
\boxed{
(1-r)\tanh!\left(\frac{r}{2\tau}\right)
\ge
r(2p+1+r).
}
\tag{32}
]

This gives a second critical friction level (\tau_c(p)).

Define

[
\eta(x):=\frac{2\tanh(x/2)}{x}.
]

Let (x_c>0) be the unique solution of

[
\boxed{
p=
\frac{x_c\eta(x_c)^2}{4[1-\eta(x_c)]}
-1+\frac{\eta(x_c)}2.
}
\tag{33}
]

Then

[
\boxed{
r_c=1-\eta(x_c),
\qquad
\tau_c(p)=\frac{1-\eta(x_c)}{x_c}.
}
\tag{34}
]

The symmetric pure phase is therefore

[
\boxed{
\tau_c(p)\le\tau<\tau_0(p)
\quad\Longrightarrow\quad
F^*=\delta_{r_p(\tau)}.
}
\tag{35}
]

For (\tau<\tau_c(p)), no positive symmetric pure equilibrium exists and the equilibrium is mixed and atomic.

Some threshold values are:

| (p=P/R) |   (\tau_c(p)) | (\tau_0(p)) |
| ------: | ------------: | ----------: |
|     (0) | (0.132116913) |       (0.5) |
|     (1) | (0.107294600) |       (1/6) |
|    (10) | (0.023485399) |      (1/42) |

This explains why the (P=1,\tau=0.1) point in Figure 11 is already mixed, while (P=10,\tau=0.1) is essentially the zero-risk equilibrium.

---

## 4. Exact two-atom equilibrium

Suppose the symmetric equilibrium support is

[
{0,x},
]

with mass (\alpha) at zero and mass (1-\alpha) at (x).

Let

[
\ell=\frac1{1+e^{-x/\tau}}.
]

For arbitrary Gaussian correlation define

[
c_\rho(x)=C_\rho(x,x),
\qquad
d_\rho(x)=1-2x+c_\rho(x),
]

and

[
q_\rho(x)=
\left.\frac{\partial C_\rho(a,x)}{\partial a}\right|_{a=x}.
]

For (-1<\rho<1), putting (z=\Phi^{-1}(x)) and

[
\beta=\sqrt{\frac{1-\rho}{1+\rho}},
]

gives

[
\boxed{
c_\rho(x)=x-2T(z,\beta),
\qquad
q_\rho(x)=\Phi(\beta z),
}
\tag{36}
]

where (T) is Owen’s (T)-function.

The four relevant payoffs are

[
\begin{aligned}
U(0,0)&=\frac12,\
U(0,x)&=1-(1-x)\ell,\
U(x,0)&=-px+(1-x)\ell,\
U(x,x)&=\frac12\left[1-c_\rho(x)\right]-px.
\end{aligned}
\tag{37}
]

Indifference between zero and (x) gives

[
\boxed{
\alpha(x)
=========

\frac{
1+c_\rho(x)+2px-2(1-x)\ell
}{
c_\rho(x)
}.
}
\tag{38}
]

For independent failures (c_0(x)=x^2).

The two derivatives required at (x) are

[
\boxed{
d_{x0}
======

-p-\ell+
\frac{(1-x)\ell(1-\ell)}{\tau},
}
\tag{39}
]

and

[
\boxed{
d_{xx}
======

-p-\frac12-\frac12q_\rho(x)
+
\frac{d_\rho(x)}{4\tau}.
}
\tag{40}
]

The positive atom is stationary exactly when

[
\boxed{
\alpha(x)d_{x0}
+
[1-\alpha(x)]d_{xx}
=0.
}
\tag{41}
]

Thus a two-atom equilibrium is obtained from a **single scalar equation** (41), followed by the checks

[
0<\alpha(x)<1
]

and

[
\alpha U(a,0)+(1-\alpha)U(a,x)\le v
\quad\forall a\in[0,1].
\tag{42}
]

At (\tau=\tau_c), equation (38) gives (\alpha=0), so the two-atom branch joins continuously to the positive pure branch.

---

## 5. Higher-atom equilibria

As (\tau) decreases, another off-support local maximum can reach the equilibrium payoff. At such a transition, a new support point (y) satisfies

[
V(y)=v,
\qquad
V'(y)=0.
\tag{43}
]

After adding it to the support, equations (23)–(25) determine the new equilibrium exactly.

For example, for

[
p=10,\qquad \tau=0.01,\qquad \rho=0,
]

the exact continuous-action equilibrium has three atoms:

[
\boxed{
x=
\left(
0,;
0.0253848276193,;
0.0536891318757
\right),
}
]

with probabilities

[
\boxed{
w=
\left(
0.31421040,;
0.26824276,;
0.41754684
\right).
}
]

It has

[
\bar r=0.0292270234679
]

and

[
u_{\mathrm{total}}
==================

# 1-\bar r^2-20\bar r

0.414605311741.
]

Substitution into (23)–(25) gives equation residuals below (3\times10^{-15}), and a dense global best-response check gives no profitable deviation to numerical precision.

For smaller (\tau), the number of atoms increases. This is why the regret-matching distribution looks progressively more like the continuous (\tau=0) density even though, for every fixed (\tau>0), the exact independent-risk equilibrium remains finite and atomic.

---

## 6. Both correlation and friction nonzero

The finite-support equilibrium equations remain the same after replacing (ab) by (C_\rho(a,b)).

The general own-action derivative is

[
\boxed{
U_a(a,b)
========

## -p-s_\tau(a-b)

[1-s_\tau(a-b)]C_a(a,b)
+
[1-a-b+C_\rho(a,b)]s_\tau'(a-b).
}
\tag{44}
]

Consequently:

* the zero-risk dominance threshold (26) is independent of (\rho);
* pure symmetric candidates solve

[
\boxed{
d_\rho(r)
=========

2\tau\left[2p+1+q_\rho(r)\right];
}
\tag{45}
]

* the candidate is an equilibrium exactly when

[
U(a,r)\le U(r,r)
\qquad\forall a\in[0,1];
\tag{46}
]

* two-atom equilibria are given by (36)–(42);
* higher finite-support branches are given by (23)–(25) using (C_\rho) and derivative (44).

For (\rho=\pm1), the copula has a kink along the diagonal, so equations involving (C_a(x,x)) are replaced by the corresponding one-sided KKT inequalities.

As a concrete combined-parameter example,

[
p=1,\qquad \tau=0.1,\qquad \rho=-0.5
]

has the pure equilibrium

[
r=0.1970129972.
]

A full best-response calculation gives its maximum exactly at (a=r). By contrast, with the same (p,\tau) but (\rho=0.5), the diagonal first-order equation has a root near (0.21624), but it is not an equilibrium: deviation to zero increases utility. This illustrates why equation (45) must always be accompanied by the global condition (46).

---

# Checks against the solver

The released experiment uses (R=1,Z=0), 500 shifted actions, and `iters=2000`; the regret-matching implementation multiplies the iteration argument by the action count, yielding (10^6) stochastic updates per parameter point. The two players use interlaced rather than identical grids. ([GitHub][2])

## Correlation-only check, (p=1,\tau=0)

An independent rerun of the released algorithm with 300 shifted actions and (600{,}000) updates gives:

| (\rho) | Analytical (\bar r) | Solver (\bar r) | Analytical total (u) | Solver total (u) |
| -----: | ------------------: | --------------: | -------------------: | ---------------: |
|   (-1) |          (0.250000) |      (0.249246) |           (0.500000) |       (0.501509) |
|    (0) |          (0.236068) |      (0.236333) |           (0.472136) |       (0.471481) |
|    (1) |          (0.213061) |      (0.212203) |           (0.426123) |       (0.428688) |

The deviations are consistent with grid spacing and finite regret.

## Pure frictional phase

For

[
p=1,\qquad \rho=0,\qquad \tau=0.12,
]

equation (30) gives

[
r=0.1328829857,
\qquad
u_{\mathrm{total}}=0.7165761408.
]

A 500-action regret-matching rerun gives

[
\bar r=0.1329285373,
\qquad
u_{\mathrm{total}}=0.7164729300.
]

Thus the solver concentrates on the predicted pure action.

## Two-atom phase and Figure 11

For

[
p=1,\qquad \rho=0,\qquad \tau=0.1,
]

equations (38)–(41) give

[
x=0.1878324074,
\qquad
\alpha=0.0557644403,
]

so

[
\bar r=(1-\alpha)x=0.1773580383,
]

and

[
u_{\mathrm{total}}=0.6138280496.
]

The corresponding released Figure 11 point is approximately

[
\bar r_{\rm solver}=0.1780528369,
\qquad
u_{\rm solver}=0.6121674417.
]

For (p=10,\tau=0.1), equation (26) gives

[
\tau_0=\frac1{42}\approx0.02381<0.1,
]

so the exact solution is

[
\bar r=0,\qquad u_{\mathrm{total}}=1.
]

The released point is

[
\bar r_{\rm solver}\approx0.00050225,
\qquad
u_{\rm solver}\approx0.989891,
]

which is precisely what one expects from the shifted grid: one player’s smallest available action is approximately (0.001), rather than zero.

Across all 72 recovered Figure 11 points, the solver’s reported utilities satisfy

[
u=1-2P\bar r-\bar r^2+\delta^2
]

to within (2.36\times10^{-4}), where (\delta) is the numerical asymmetry between the two players’ mean risks. Thus the payoff computation is correct; the remaining differences are equilibrium-approximation errors from discretization and finite regret matching. The paper itself describes these as approximate equilibria and evaluates them with NashConv rather than claiming exact convergence. ([arXiv][1])

[Download the recovered Figure 11 solver data](sandbox:/mnt/data/figure11_recovered_solver_points_and_formula.csv)

## Final analytical characterization

With (p=P/R):

[
\boxed{
\begin{array}{ll}
\tau=0,\ \rho\text{ arbitrary}:&
f=p(I-K_{\rho,h})^{-1}D_\rho^{-1},
\quad
\int f=1;[2mm]
\tau>0,\ \rho=0:&
\text{every equilibrium is finite and satisfies (23)–(25)};[2mm]
\tau\ge \dfrac{1}{2(2p+1)}:&
F^*=\delta_0;[4mm]
\tau_c(p)\le\tau<
\dfrac{1}{2(2p+1)},\ \rho=0:&
F^*=\delta_{,1+\tau-\sqrt{\tau^2+4\tau(p+1)}};[4mm]
\tau<\tau_c(p):&
\text{finite atomic mixture determined by (23)–(25)};[2mm]
\rho\ne0,\ \tau>0:&
\text{use the same atomic equations with }C_\rho
\text{ and derivative (44).}
\end{array}
}
]

This is the analytical solution underlying the numerical phase diagrams: correlation preserves the continuous rank-order equilibrium when (\tau=0), whereas positive friction regularizes the payoff discontinuity and replaces the continuous distribution by pure or finitely atomic equilibria.

[1]: https://arxiv.org/pdf/2305.18941 "A Game of Competition for Risk"
[2]: https://github.com/louisabraham/cfrgame/blob/main/experiments.py "https://github.com/louisabraham/cfrgame/blob/main/experiments.py"
