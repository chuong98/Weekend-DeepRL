# Soft Actor-Critic 
> Haarnoja, T., Zhou, A., Hartikainen, K., Tucker, G., Ha, S., Tan, J., Kumar, V., Zhu, H., Gupta, A., Abbeel, P. and Levine, S., 2018. Soft actor-critic algorithms and applications. arXiv preprint arXiv:1812.05905.

### 1. Motivation
+ Q-Learning Actor-Critic methods, such as DDPG or TD3, are able to learn efficiently from the past samples using experience replay buffers. However, the problem with these methods is that they are very sensitive to hyperparameters and require a lot of tuning to get them converge.
+ Soft Actor-Critic address the convergence problem by:
  + Not only seeking to maximize the lifetime rewards.
  + But also maximize the **entropy** of the Policy. 
+ **What is "Entropy"?** 
  + In a simple explaination, "entropy" means "unpredictable". For example, if a variable always takes a single value, it has 0 entropy. But if it is randomly sampled from a uniform distribution, it has high entropy since it is very unpredictable. 
  + Formally, if `x` is a random variable with a distribution $P$, then Entropy is computed as:
  $$ H(P) = E_{x \sim P} [-\log P(x)]$$
+ **Why do we want our policy to have high "Entropy"?** 
  + During learning, we want to encourage the exploration, we don't want the agent trapped in a local optimum, e.g. short term rewards.
  + This in turn also helps improve the estimation of Q-function, i.e Critic Network. Because of the nature of boostrap temporal learning, the Critic network tends to be over-confident estimation.

### 2. Soft Actor-Critic
+ From the motivation, the learning objective is modified to
    $$ \pi^* = \argmax_{\pi} E_{a \sim \pi} [ \sum_{t=0}^{\infty} \gamma^t(\underbrace{r_t + \alpha H(\pi(\cdot|s_t))}_{\large r^+_t})] $$ 
  where $H(\pi)$ is the entropy of the Policy $\pi$, and $\alpha >0$ is the trade-off coefficient. 

     What the Objective equation says is that, **the agent gets a bonus reward** at each time step, propotional to the entropy of the policy at that timestep.  
+ As a result, the value function $V$, the quality function $Q$, and the Bellman equation are also modified to reflect the object learning. However, if we denote $r^+_t = r_t + \alpha H(\pi(\cdot|s_t))$ as the new (bonused) reward, and use it instead of $r$, then we can use old formulars as in TD3. 
+ What is similar/different with TD3 ?
  + Critic-Networks: 
    + SAC still uses the twin Q-networks, exactly as in TD3.
    + SAC uses $r^+$ instead of $r$.
    + SAC has only 1 actor-network (no actor-target), and we use it to compute the $q_{target}$. 
  + Policy:
    + TD3 uses $Q_1(s,a)$ to compute the policy loss, but SAC uses $\min (Q_1,Q_2)$ instead.  
    + TD3 uses a **deterministic** action function: $a=A_\phi(s)$, thus it needs add Gaussian noise to improve the exploration $a \leftarrow a + N(0,\sigma)$.
    + SAC uses a **stochasitic** action function: $a \sim \pi_{\phi}(a|s)=N(a|\mu,\sigma)$, which is a Gaussian distribution, where $(\mu,\sigma)$ are predicted by the actor-network  $A_{\phi}(s)$. We relizes the action sampling by a **squashed Gaussian**:
      $$ \tilde{a}(s,\xi) = \tanh(\mu + \sigma \xi ), \hspace{1cm} \xi \sim N(0,1) $$       
      where $tanh$ is **a squashing function** mapping the action to a normalzied range $(-1,1)$, and $\xi$ is the random variable to sample the action. This formulation allows us to update the network by gradient descent, and is called the **reparameterized trick**. Note, we use $\tilde{a}$ to denote it is obtained by **randomly sampling (stochasitic)**.
  + TD3 delays updating the Action Network and Target networks, but SAC does not (we can try).

### 3. Implementation
#### 3.1. Key Equations for implementing:
+ Critic-Network: Let $Q_{\theta_1}(s,a)$ and $Q_{\theta_2}(s,a)$ be the twin critic-networks, and $Q_{\theta_{T1}}(s,a)$ and $Q_{\theta_{T2}}(s,a)$ are their corresponding target networks, respectively. We train the critic-networks to optimize the loss function:
  $$ Loss_Q = (Q_{\theta_1}(s,a) - q_{target})^2 + (Q_{\theta_2}(s,a) - q_{target})^2$$
  $$ \text{where} \hspace{1cm}  q_{target} = r + \gamma*(Q_{\min}(s',\tilde{a}') + \alpha H(\pi_\phi(\tilde{a}|s'))$$,
  $$ \text{with} \hspace{1cm} Q_{\min} (s',\tilde{a}') = \min[ Q_{\theta_{T1}}(s',\tilde{a}'),Q_{\theta_{T2}}(s',\tilde{a}')]$$ 
+ Actor Network: Let $A_\phi(s)$ be the actor-network predicting $(\mu,\sigma)$ of the Gaussian policy : 
  $$ \pi_{\phi}(u) = N(u|\mu,\sigma) = \frac{1}{\sigma\sqrt{2\pi}} \exp(-\frac{1}{2}(\frac{u-\mu}{\sigma})^2)$$
  and the action $\tilde{a} = \tanh(u)$, where $u \sim \pi_\phi(u)$. We train the actor-network to optimize the loss function:
  $$ Loss_A = - Mean[Q_{\min}(s,\tilde{a}) + \alpha H( \pi_{\phi}(\tilde{a}|s)) ]$$ 

+ Implementing entropy $H(\pi_\phi(\tilde{a}|s))$: is quite tricky, so we break it down into two parts:
  + Entropy of a random variable $u$ sampled from Gaussian distribution:
    $$H(\pi_\phi(u|s)) = -\log \pi_\phi(u|s) = - \log N(u|\mu,\sigma)$$
    $$\log N(u|\mu,\sigma)= -\frac{1}{2} (\frac{u-\mu}{\sigma})^2 - \log \sigma - \log{\sqrt{2\pi}}$$
  + Entropy of a random variable $a$ sampled from Squashed Gaussian distribution: Since $a=\tanh(u), u \sim N(u|\mu,\sigma)$, the density function of $a$ [(explaination)](https://math.stackexchange.com/questions/3108216/change-of-variables-apply-tanh-to-the-gaussian-samples) is:
    $$\pi_\phi(a|s) = N(u|\mu,\sigma) |\det(\frac{da}{du})|^{-1} $$
    Since the Jacobian matrix $da/du = diag(1 - \tanh^2(u))$ is diagonal, the log-likelihood has a simple form:
    $$\log \pi_\phi(a|s) = \log N(u|\mu,\sigma) - \sum_{i=1}^{D}\log(1 - \tanh^2(u_i))$$
    where $D$ is the dimension of action space. By substituting $\tanh(u)= \large \frac{1-e^{-2u}}{1+e^{-2u}}$, we get:
    $$\log(1 - \tanh^2(u))= \log \frac{4e^{-2u}}{(1 + e^{-2u})^2} = 2(\log2 - u -\log(1+e^{-2u}))$$ 
#### 3.2 Coding tricks:
  + In practice , we use the actor network to predict $y_\sigma=\log(\sigma)$, and then take exponent to obtain $\sigma=\exp(y_\sigma)$. This is because, 
   $\log(\sigma)$ can be any value $(-\infty, +\infty)$ while $\sigma$ must be positive. It is also more numerical stable, since taking exponent is easier than $\log$.
    ```python
    mu, log_sigma = actor_network(s)
    sigma = log_sigma.exp()
    ```
  + Pytorch provides a nice function to sample from distribution (with gradient backward):
    ```python 
    from torch.distribution import Normal 
    pi_dist= Normal(mu, sigma)
    u = pi_dist.sample()  # cann't do gradient backward.
    u = pi_dist.rsample() # can do gradient backward.
    a = torch.tanh(u)     # action is squashed to (-1,1)
    ```
  + Computing entropy is implemented as:
    ```python
    log_prob_a = pi_dist.log_prob(u).sum(axis=-1)
    log_prob_a -= (2*(np.log(2) - u - F.softplus(-2*u))).sum(axis=1)
    entropy = - log_prob_a
    ```

### 4. What make SAC work?
+ Similar to TD3, it uses twin critic-networks to improve the estimation of Q value.
+ It uses stochasitic action function to allow exploration by its formulation.
+ The key to add entropy of the policy as an bonus reward to avoid being trapped in the local minimal.
+ Tuning the temperature $\alpha$ is challenging. In late version, the authors address this problem by proposing a mechanism to auto-tune $\alpha$

