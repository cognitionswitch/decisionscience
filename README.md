# Prospect Theory and Cumulative Prospect Theory Agent Demo

The PTAgent and CPTAgent classes reproduce patterns of choice behavior described by Kahneman & Tverski's survey data in their seminal papers on Prospect Theory and Cumulative Prospect Theory. These classes expresses valuations of single lottery inputs, or express preferences between two lottery inputs. To more explicitly describe these agent classes, we define the following:

1. $(x_1, p_1; \cdots; x_n, p_n)$: a lottery offering outcome $x_1$ with probability $p_1$, ..., outcome $x_n$ with probability $p_n$. 
2. $v(x)$: the internal representation of the value of an outcome $x$ to an instance of a PTAgent.
3. $\pi(p)$: the internal representation of a probability $p$ to an instance of a PTAgent. 
4. $V(x_1, p_1; \cdots; x_n, p_n)$: a lottery valuation function.

#### **Prospect Theory Agent**

The PTAgent class reflects the lottery valuation function of Prospect Theory described in Kahneman & Tverski (1979). Generally, the lottery valuation function operates as follows: 

$$V(x_1, p_1; \dots; x_n, p_n) = v(x_1) \times \pi(p_1) + \cdots + v(x_n) \times \pi(p_n) \tag{1a}$$

However, under certain conditions the lottery valuation function is operates under a different formulation. These conditions are:

1. When the lottery contains exactly two non-zero outcomes and one zero outcome relative to a reference point, with each of these outcomes occuring with non-zero probability; ie., $p_1 + p_2 + p_3 = 1$ for $x_1, x_2 \in \lbrace x | x \ne 0 \rbrace$ and $x_3=0$.
2. When the outcomes are both positive relative to a reference point or both negative relative to a reference point. Explicitly, $x_2 < x_1 < 0$ or $x_2 > x_1 > 0$.

When a lottery satisfies the conditions above, the lottery valuation function becomes:

$$V(x_1, p_1; x_2, p_2) = x_1 + p_2(x_2 - x_1) \tag{1b}$$

Since the original account of prospect theory does not explicitly describe the value function or weighting function, the value function uses the same function proposed in Tverski & Kahneman (1992):

 $$v(x) = \begin{equation}
\left\{ 
  \begin{aligned}
    x^\alpha& \;\; \text{if} \, x \ge 0\\
    -\lambda (-x)^\beta& \;\; \text{if} \, x \lt 0\\
  \end{aligned}
  \right.
\end{equation} \tag{2}$$

While the weighting function uses a form described here: https://sites.duke.edu/econ206_01_s2011/files/2011/04/39b-Prospect-Theory-Kahnemann-Tversky_final2-1.pdf.

$$\pi(p) = exp(-(-ln(p))^\gamma) \tag{3}$$

#### **Cumulative Prospect Theory Agent**

The CPTAgent class reflects the lottery valuation function, value function, and weighting function described in Tverski & Kahneman (1992). The CPTAgent class also incorporates capacities as described in this same paper. For Cumulative Prospect Theory, outcomes and associated probabilities include the attribute of *valence* that reflects whether the realization of an outcome would increases or decreases value from a reference point of the agent. 

The value function for positive and negative outcomes is shown in equation 2 above.

For probabilities $p$ associated with positive valence outcomes, the *capacity* function is expressed as:
$$w^{+}(p) = \frac{p^\gamma}{\left(p^\gamma+(1-p)^\gamma) \right)^{1/ \gamma}} \tag{4a}$$

For probabilities $p$ associated with negative valence outcomes, the capacity function is expressed similarly as:
$$w^{-}(p) = \frac{p^\delta}{\left(p^\delta+(1-p)^\delta) \right)^{1/ \delta}} \tag{4b}$$

In order to compute a weight for the $i^{th}$ outcome with positive valence, a difference of cumulative sums is computed as follows:

$$\pi^{+}(p_i) = w^{+}(p_i + \cdots + p_n) - w^{+}(p_{i+1} + \cdots + p_n), \; 0 \le x_i < \cdots < x_n \tag{5a}$$

Similarly, computing a weight for the $j^{th}$ outcome with negative valence:

$$\pi^{-}(p_j) = w^{-}(p_j + \cdots + p_m) - w^{-}(p_{j+1} + \cdots + p_m), \; 0 \gt x_j > \cdots > x_m \tag{5b}$$

Lottery valuations for Cumulative Prospect Theory are then computed in a similar manner as Prospect Theory (equation 1a). 

---

## Choice Behavior for Lotteries

#### **Normative Choice Behavior**

Specification of the following parameters leads to an agent that chooses lotteries according to Expected Utility Theory:
- $\alpha = \beta = 1$
- $\gamma = \delta = 1$
- $\lambda = 1$

#### **Descriptive Choice Behavior**

When $\alpha, \beta, \gamma, \delta$ take values on the interval $(0, 1)$, and when $\lambda > 1$, lottery valuation functions with constituent value and weighting functions show patterns of choice that better approximate empirical choice behavior than those predicted by normative choice behavior.

#### **Notation**

To illustrate functionality of the PTAgent and CPTAgent classes, we denote an outcome and its associated probability as a tuple $(G_1, p_1)$ and $(L_1, p_1)$, where $G_1$ is used to denote gains and $L_1$ denotes losses. A lottery is a set of gains and/or losses with associated probabilities: $[(L_1, p_1), \cdots, (G_n, p_n)]$, where $\sum p_i = 1$. A preference between two prospect, for example "A is prefered to B", is denoted as $A > B$.  

The following instance of PTAgent uses function parameters estimated in Tverski & Kahneman (1992). These parameters are sufficient to replicate observed modal choices between prospects in (Kahneman & Tverski, 1992) and (Tverski & Kahneman, 1992).

---

### Decision Anomalies

The demonstrations below show instances of the PTAgent class exhibiting the same choice anomalies discussed in Kahneman & Tverskies seminal paper on Prospect Theory (1979).


```python
from cpt_agent import PTAgent
```


```python
pt = PTAgent(alpha=0.88, gamma=0.61, lambda_=2.25)
```

### The certainty effect

The certainty effect demonstrates that reducing the probability of outcomes from certainty has larger effects on preferences than equivalent reductions from risky (ie., non-certain) outcomes. Problems 1 and 2 illustrate this effect for absolute reductions in probabilities and problems 3 and 4 show this effect for relative reductions in probabilities. 

- Problem 1: $[(G_1, p_1), (G_2, p_2), (0, p_3)] < [(G_2, 1)]$
- Problem 2: $[(G_1, p_1), (G_2, 0), (0, p_3)] > [(G_2, 1-p_2)]$

Subtracting probability $p_2$ of outcome $G_2$ from both options in problem 1 leads to a preference reversal in problem 2.


```python
# Problem 1
lottery_1A = {'outcome':[2500, 2400, 0], 'probability':[0.33, 0.66, 0.01]}
lottery_1B = {'outcome':[2400], 'probability':[1]}

pt.choose(lottery_1A, lottery_1B)
```




    {'lottery2': {'outcome': [2400], 'probability': [1]}}




```python
# Problem 2
lottery_2C = {'outcome':[2500, 0], 'probability':[0.33, 0.67]}
lottery_2D = {'outcome':[2400, 0], 'probability':[0.34, 0.66]}

pt.choose(lottery_2C, lottery_2D)
```




    {'lottery1': {'outcome': [2500, 0], 'probability': [0.33, 0.67]}}



Scaling probabilities of risky outcome $G_1$ and certain outcome $G_2$ by $p'$ in problem 3 leads to a preference reversal in problem 4. This preference reversal violates the substitution axiom of Expected Utility Theory.

- Problem 3: $[(G_1, p_1), (0, 1-p_1)] < [(G_2, 1)]$
- Problem 4: $\left[\left(G_1, p_1\cdot p'\right), \left(0, \frac{1-p_1}{p'}\right)\right] > [(G_2, p'), (0, 1-p')]$


```python
# Problem 3
lottery_3A = {'outcome':[4000, 0], 'probability':[0.8, 0.2]}
lottery_3B = {'outcome':[3000], 'probability':[1]}

pt.choose(lottery_3A, lottery_3B)
```




    {'lottery2': {'outcome': [3000], 'probability': [1]}}




```python
# Problem 4
lottery_4C = {'outcome':[4000, 0], 'probability':[0.2, 0.8]}
lottery_4D = {'outcome':[3000, 0], 'probability':[0.25, 0.75]}

pt.choose(lottery_4C, lottery_4D)
```




    {'lottery1': {'outcome': [4000, 0], 'probability': [0.2, 0.8]}}



### The reflection effect

The reflection effect demonstrates that altering outcomes by recasting prospects from the domain of gains to losses will correspondingly alter decision behavior from risk-aversion to risk-seeking. Since the reflection effect highlights preferences characterized as risk-seeking in the loss domain, the effect disqualifies risk-aversion as a general principle for explaining the certainty effect above. 

- Problem 3: $[(G_1, p_1), (0, 1-p_1)] < [(G_2, 1)]$
- Problem 3': $[(-G_1, p_1), (0, 1-p_1)] > [(-G_2, 1)]$


```python
# Problem 3'
lottery_3A_, lottery_3B_ = lottery_3A.copy(), lottery_3B.copy()
lottery_3A_.update({'outcome':[-g for g in lottery_3A_['outcome']]})
lottery_3B_.update({'outcome':[-g for g in lottery_3B_['outcome']]})

pt.choose(lottery_3A_, lottery_3B_)
```




    {'lottery1': {'outcome': [-4000, 0], 'probability': [0.8, 0.2]}}



- Problem 4: $\left[\left(G_1, p_1\cdot p^{*}\right), \left(0, \frac{1-p_1}{p^{*}}\right)\right] > [(G_2, p^{*}), (0, 1-p^{*})]$
- Problem 4': $\left[\left(-G_1, p_1\cdot p^{*}\right), \left(0, \frac{1-p_1}{p^{*}}\right)\right] < [(-G_2, p^{*}), (0, 1-p^{*})]$


```python
# Problem 4'
lottery_4C_, lottery_4D_ = lottery_4C.copy(), lottery_4D.copy()
lottery_4C_.update({'outcome':[-g for g in lottery_4C_['outcome']]})
lottery_4D_.update({'outcome':[-g for g in lottery_4D_['outcome']]})

pt.choose(lottery_4C_, lottery_4D_)
```




    {'lottery2': {'outcome': [-3000, 0], 'probability': [0.25, 0.75]}}



### Risk Seeking in Gains, Risk Aversion in Losses

In addition to violations of the substitution axiom, scaling probabilities of lotteries with a result of highly improbable outcomes can induce risk seeking in gains, and risk aversion in losses. While these characteristics of choice behavior are not violations of normative theories of choice behavior, they contrast with more typical observations of risk aversion in gains and risk seeking in losses for outcomes that occur with stronger likelihood. In the domain of gains, risk seeking for low probability events seems to correspond to the popularity of state lotteries.

- Problem 7: $[(G_1, p_1), (0, 1-p_1)] < [(G_2, p_2), (0, 1-p_2)]$
- Problem 8: $\left[\left(G_1, p_1\cdot p'\right), \left(0, \frac{1-p_1}{p'}\right)\right] > \left[\left(G_2, p_2\cdot p'\right), \left(0, \frac{1-p_2}{p'}\right)\right]$


```python
# Problem 7
lottery_7A = {'outcome':[6000, 0], 'probability':[0.45, 0.55]}
lottery_7B = {'outcome':[3000, 0], 'probability':[0.9, 0.1]}

pt.choose(lottery_7A, lottery_7B)
```




    {'lottery2': {'outcome': [3000, 0], 'probability': [0.9, 0.1]}}




```python
# Problem 8
lottery_8C = {'outcome':[6000, 0], 'probability':[0.001, 0.999]}
lottery_8D = {'outcome':[3000, 0], 'probability':[0.002, 0.998]}

pt.choose(lottery_8C, lottery_8D)
```




    {'lottery1': {'outcome': [6000, 0], 'probability': [0.001, 0.999]}}



Just as Prospect Theory accounts for risk seeking in gains for low probability events, the theory also accounts for risk aversion in the domain of losses when outcomes occur very infrequently. Risk aversion in the domain of losses seems to match well with consumer purchase of insurance products.


```python
# Problem 7'
lottery_7A_, lottery_7B_ = lottery_7A.copy(), lottery_7B.copy()
lottery_7A_.update({'outcome':[-g for g in lottery_7A_['outcome']]})
lottery_7B_.update({'outcome':[-g for g in lottery_7B_['outcome']]})

pt.choose(lottery_7A_, lottery_7B_)
```




    {'lottery1': {'outcome': [-6000, 0], 'probability': [0.45, 0.55]}}




```python
# Problem 8'
lottery_8C_, lottery_8D_ = lottery_8C.copy(), lottery_8D.copy()
lottery_8C_.update({'outcome':[-g for g in lottery_8C_['outcome']]})
lottery_8D_.update({'outcome':[-g for g in lottery_8D_['outcome']]})

pt.choose(lottery_8D_, lottery_8D_)
```




    {'lottery2': {'outcome': [-3000, 0], 'probability': [0.002, 0.998]}}



### Probabilistic Insurance

Kahneman & Tverski discuss another frequent choice anomalie called *probabilistic insurance*. To demonstrate choice behavior matching this anomalie, we first need to find a point of indifference reflecting the following relationship between current wealth $w$ and the cost of an insurance premium $y$ against a potential loss $x$ that occurs with probability $p$:

$$pu(w-x) + (1-p)u(w) = u(w-y) \tag{6}$$

That is, we are finding the premium $y$ for which a respondent is ambivelant between purchasing the insurance against loss $x$, and simply incurring the loss $x$ with probability $p$. Kahneman & Tverski introduce an insurance product called probabilistic insurance whereby the consumer only purchases a portion $r$ of the premium $y$. If the event leading to loss actually occurs, the purchaser pays the remainder of the premium with probability $r$, or is returned the premium and suffers the loss entirely with probability $1-r$. 

$$(1-r) p u(w-x) + rpu(w-y) + (1-p)u(w-ry) \tag{7}$$

Kahneman & Tverski show that according to Expected Utility Theory, probabilistic insurance is generally preferred to either a fully insured product $u(w-y)$ or a loss $x$ with probability $p$ (under the assumption of ambivalence described above). In surveys, however, respondents generally show a strong preference against probabilistic insurance.


```python
# Problem 9
premium = 1000
asset_am = 6000
loss = 5000
prob_loss = 0.06925

lottery_9A = {'outcome':[asset_am - premium], 'probability':[1]}
lottery_9B = {'outcome':[asset_am - loss, asset_am], 'probability':[prob_loss, 1-prob_loss]}
```


```python
pt.evaluate(lottery_9A)
```




    1799.2586689124155




```python
pt.evaluate(lottery_9B)
```




    1799.313595333693




```python
# Problem 10
r = 0.94

lottery_10A = {'outcome':[asset_am - loss, asset_am - premium, asset_am - r*premium], 
               'probability':[(1-r)*prob_loss, r*prob_loss, (1-prob_loss)]}
```


```python
pt.choose(lottery_9B, lottery_10A)
```




    {'lottery1': {'outcome': [1000, 6000], 'probability': [0.06925, 0.93075]}}



### Cumulative Prospect Theory

Kahneman & Tverski modified their original account of Prospect Theory with Cumulative Prospect Theory (1990). The CPTAgent exhibits the same choice behavior shown by the PTAgent for each of the problems considered above. Additionally, the cumulative features of the weighting function better demonstrates the choice patterns of respondents when considering probabilistic insurance, namely, the preference against probabilistic insurance seems to hold under a broader range of probabilities $r$. 


```python
from cpt_agent import CPTAgent
```


```python
cpt = CPTAgent(alpha=0.88, gamma=0.61, lambda_=2.25)
```


```python
# Problem 11
r = 0.73

lottery_10B = {'outcome':[asset_am - loss, asset_am - premium, asset_am - r*premium], 
               'probability':[(1-r)*prob_loss, r*prob_loss, (1-prob_loss)]}

cpt.choose(lottery_9A, lottery_10B)
```




    {'lottery1': {'outcome': [5000], 'probability': [1]}}


