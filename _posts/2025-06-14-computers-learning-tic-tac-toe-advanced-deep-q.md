---
layout: post
title:  "Computers learning Tic-Tac-Toe Pt. 4: Towards Rainbow DQN"
date:   2025-06-14
description: Three add-ons that stabilise and speed up deep Q-learning - Double DQN, Dueling DQN and Prioritised Experience Replay
---

<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>



<style>
  .sweep-selector {
    margin: 1em 0;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    text-align: center;
  }

  .sweep-selector label {
    font-weight: 600;
    font-size: 1rem;
    color: #1d1d1f;
    margin-bottom: 0.5em;
    display: block;
  }

  .sweep-selector select {
    width: 100%;
    max-width: 300px;
    padding: 0.6em 1em;
    font-size: 1rem;
    font-weight: 500;
    color: #1d1d1f;
    background-color: #f2f2f7;
    border: 1px solid #d1d1d6;
    border-radius: 12px;
    appearance: none;
    -webkit-appearance: none;
    transition: border 0.2s ease, box-shadow 0.2s ease;
    background-image: url("data:image/svg+xml,%3Csvg fill='%23666' viewBox='0 0 24 24' width='18' height='18' xmlns='http://www.w3.org/2000/svg'%3E%3Cpath d='M7 10l5 5 5-5z'/%3E%3C/svg%3E");
    background-repeat: no-repeat;
    background-position: right 1em center;
    background-size: 1em;
  }

  .sweep-selector select:focus {
    border-color: #007aff;
    outline: none;
    box-shadow: 0 0 0 3px rgba(0, 122, 255, 0.3);
  }
</style>

<p class="intro"><span class="dropcap">I</span>n the <a href="https://kunkelalexander.github.io/blog/computers-learning-tic-tac-toe-optimisation/">previous post</a> we optimised our single-network Deep Q-Network (DQN) agent.
Today we level-up that agent with three classic extensions that address DQN’s biggest pain points: Double DQN, Dueling DQN and Prioritised Experience Replay.</p>


Before diving into improvements, we also **add vanilla DQN** — i.e. the correct implementation with **two separate networks** — as a baseline reference. Our earlier agent used a simplified setup where the target and online networks were identical ($$ Q_\theta = Q_{\theta^-} $$), which is known to lead to instability. We now evaluate how much benefit we get by simply introducing the second network as originally proposed in DQN.

We study four setups in more detail:

1. **Vanilla DQN** – two separate networks; a stable target using $$ Q_{\theta^-} $$
2. **Double DQN** – fixes systematic over-estimation of action values.
3. **Dueling DQN** – separates *what* a state is worth from *which* action is best.
4. **Prioritised Experience Replay (PER)** – replays the transitions that matter most.

Each idea is independent and inexpensive to bolt on, yet in combination they underpin most modern DQN-style agents (e.g. *Rainbow*).

### Vanilla DQN – two networks for stability

Our original DQN agent used a single neural network for both action selection and temporal difference target (TD-target) calculation — i.e., we set the target and online networks to be the same ($$ Q_\theta = Q_{\theta^-} $$). This made the implementation simple but deviated from the original design proposed by [Mnih et al. (2015)](https://www.nature.com/articles/nature14236).

Vanilla DQN corrects this by maintaining **two separate networks**:

- The **online network** $$ Q_\theta(s, a) $$ is updated at every training step via gradient descent.
- The **target network** $$ Q_{\theta^-}(s, a) $$ is updated less frequently. Two common strategies are a **hard update** where the target network's weights are are copied directly from the online network every N steps or a **soft update** or **Polyak averaging** where the weights of the target network are blended with the online network's weights using a small constant $$\tau$$ as $$\theta^- \leftarrow (1 - \tau) \cdot \theta +  \tau \cdot \theta^-$$.

This stabilises learning by preventing the TD-target from shifting too rapidly. Concretely, the target used in the loss is:

$$
y_t = r_{t+1} + \gamma \max_{a'} Q_{\theta^-}(s_{t+1}, a')
$$

Only this change is required in code — but it makes a significant difference in convergence stability for environments with noisy rewards or longer horizons. In our Tic-Tac-Toe setting, the benefit is smaller, but we include it as a baseline for the later extensions. You can find a comparison of the influence of these update strategies below. In general, I do not find that the double network architecture improves the performance of our Tic-Tac-Toe agent but this is probably to be expected given that rewards are not sparse.

<!-- Elegant Sweep Variable Selector -->
<div class="sweep-selector">
  <select id="plotSelector" onchange="updateImage()">
    <option value="target_update_freq">Hard Update: Frequency</option>
    <option value="target_update_tau">Soft Update: Tau</option>
  </select>
</div>


<!-- Display the selected plot -->
<figure>
<img id="plotImage" src="{{ site.baseurl }}/assets/img/tictactoe-python/vanilla_dqn_target_update_freq_sweep.png" width="100%" alt="Parameter sweep Plot"/>
  <figcaption>
 <strong>Figure 1</strong>: Performance and training loss of vanilla DQN agent against random minmax agent as a function of target network update strategy. The parameter baseline is set as follows: 3000 training episodes, evaluation every 100 episodes across 100 games, a discount factor of 0.8, learning rate of 0.01 without decay, and initial exploration rate of 1.0 with exponential decay of 0.01 per game down to 0.1. The agent uses a batch size of 128, a replay buffer of size 10,000 with a minimum of 1,000 experiences before training, and two gradient updates per training step. The agents only take legal actions. Shaded areas show standard deviation of draw rate across ten runs with different random seeds. The algorithm uses a network with a single hidden layer with 128 neurons.
</figcaption>
</figure>


<script>
  function updateImage() {
    const variable = document.getElementById("plotSelector").value;
    const imagePath = `{{ site.baseurl }}/assets/img/tictactoe-python/vanilla_dqn_${variable}_sweep.png`;
    const img = document.getElementById("plotImage");
    img.src = imagePath;
    img.alt = `${variable} Sweep Plot`;
  }
</script>


### Double (Dual) DQN – tame the over-estimation bias

Vanilla DQN uses the *same* network to both **select** and **evaluate** the action inside the TD‑target
$$y_t \;=\; r_{t+1} + \gamma \,\max_{a'} Q_{\theta^-}(s_{t+1}, a').$$
Because the `max` and the value share parameters, positive noise is amplified, leading to overly optimistic Q‑values.

Fortunately, <a href="https://arxiv.org/pdf/1509.06461/">van Hasselt et al. (2015)</a> propose a simple solution: We can **decouple selection from evaluation by keeping two separate neural networks**: an online network $$Q_\theta$$ and a target network $$ Q_{\theta^-} $$. Empirically, the bias disappears and learning becomes more stable – especially in sparse‑reward games.

The online network chooses actions and the target network is used to evaluate how good they were.

$$
a^* = \arg\max_{a'} Q_{\theta}(s_{t+1}, a') \quad\text{(online network chooses)}
$$
$$
y_t = r_{t+1} + \gamma \, Q_{\theta^-}(s_{t+1}, a^*) \quad\text{(target network evaluates).}
$$

Only this one‑line change is required in the learning update. While the performance of the network remains similar for Tic-Tac-Toe, I do detect a difference in the Q-tables shown in the figure below. As expected, the values in the Q-table reconstructed from the target network trained with Double DQN tend to be slightly lower than those of the target network reconstructed from Vanilla DQN


<figure>
  <iframe src="{{ site.baseurl }}/assets/img/tictactoe-python/17_vanilla_dqn_q_table.html" width="100%" height="600" frameborder="0"></iframe>
  <iframe src="{{ site.baseurl }}/assets/img/tictactoe-python/18_double_dqn_q_table.html" width="100%" height="600" frameborder="0"></iframe>
  <figcaption><strong>Figure 2</strong>: Q-table of a Vanilla DQN agent (top) and Double DQN agent (bottom) trained against a deterministic minimax opponent. The Q-table reconstructed from the Double DQN agent seems to be slightly less optimistic than that of the Vanilla DQN agent.</figcaption>
</figure>


### Duelling DQN – separate value from advantage

The introduction of duelling networks by <a href="https://arxiv.org/abs/1511.06581">Wang et al. (2016)</a> was guided by the realisation that on many steps the choice of action barely matters (think: standing still in *Pong* between bounces).
Yet the vanilla Q‑network must still back‑propagate a distinct value for every one of those near‑equivalent actions.
They proposed a network that first produces a **state‑value** $$V_\phi(s)$$ and an **advantage** vector $$A_\psi(s,a)$$; it then recombines them into Q‑values:

$$
Q(s,a) \;=\; V_\phi(s)
\;+\; \Bigl(A_\psi(s,a) - \tfrac{1}{|\mathcal A|}\sum_{a'} A_\psi(s,a')\Bigr).
$$

This “dueling head” lets the agent learn *how good a state is* even before it knows which action is best, speeding up credit assignment and generalisation. The network itself schematically looks as shown below: The input board state usually flows through one or several layers before being split up into the state-value and the advantage vector, possibly followed by more layers in both streams, before being aggregated according to the above equation and then sent to the output layer.

<!-- Display the selected plot -->
<figure>
<img id="plotImage" src="{{ site.baseurl }}/assets/img/tictactoe-python/19_dueling_dqn_schema.png" width="100%" alt="Duelling network"/>
  <figcaption>
  <strong>Figure 3</strong>: Schematic representation of a duelling network with data flowing from left to right. The duelling network has two streams
to separately estimate (scalar) state-value V and the advantages A for
each action; the aggregation module implements Q = V + (A - mean(A)) to
combine them. Optional additional hidden layers are omitted in this visualisation. Created with <a href="https://github.com/paulgavrikov/visualkeras">visualkeras</a> and manual annotations.
</figcaption>
</figure>

In my DQN implementation, I choose to represent both the value and advantage streams using two hidden layers with 128 neurons, each. I wondered whether the more complex network architecture would change the ideal number of neurons in these layers, but the parameter sweeü shown below suggests that fewer neurons mean worse performance.


<!-- Display the selected plot -->
<figure>
<img id="plotImage" src="{{ site.baseurl }}/assets/img/tictactoe-python/duelling_dqn_hidden_layer_sweep.png" width="100%" alt="Parameter sweep Plot"/>
  <figcaption>
  <strong>Figure 4</strong>: Performance and training loss of duelling DQN agent against random minmax agent as a function of hidden layer size where the size of all hidden layers (shared trunk, value stream, advantage stream) is set equal. Hyperparameters same as Figure 1.
</figcaption>
</figure>

What is interesting about the duelling netowrk architecture that we are training two separate outputs that we can query separately to study what they encode. Figure 5 shows both the learned state and advantage values.

<figure>
  <iframe src="{{ site.baseurl }}/assets/img/tictactoe-python/20_duelling_dqn_v_a_table.html" width="100%" height="600" frameborder="0"></iframe>
  <figcaption><strong>Figure 5</strong>: Advantage and value streams of a duelling DQN agent after 10,000 rounds of training against a deterministic minimax opponent.</figcaption>
</figure>

Looking at the state values we see the following:
- The agent predicts the **highest V-value for state 25 where it achieves a draw regardless of the next move**. The V-value is close to 0 and the A-vector is close to 0 as well. Therefore, the remaining actions lead to Q-values close to 0 following Q = V + (A - mean(A)).
- **State 0 has a high V-value** of around -0.5. Playing **center is the move with the highest A-value** of around 0.5. The mean advantage, again, is close to zero as for all states shown here. Therefore, the Q-value of the center move is close to 0 - suggesting it can lead to a draw.
- There are a number of moves with **high and low A-values in state 10** that has the third-highest V-value of around -0.5. Actions that lead to immediate defeat - bottom right, bottom center and center right have low A-values of around -0.5. **The only legal move that avoids defeat - center left - has the highest action value** translating into a Q-value of around 0.

These considerations show that the Duelling DQN agent has learned a sensible representation that separates how good a state is from which action is best. On Tic-Tac-Toe it achieves performance similar to the other DQN approaches I have introduced before. Please refer to the notebook for more plots.


### Prioritised Experience Replay – learn more from what surprises you


In standard (uniform) replay, every past transition is equally likely to be sampled for training. But not all transitions are equally “useful”: A transition where your network’s Q-estimate is very wrong, i.e. it shows large temporal-difference errors (TD errors), carries more learning signal than one it already predicts well. <a href="https://arxiv.org/abs/1511.05952">Schaul et al. (2016)</a> introduced **Prioritised Experience Replay (PER)** to replay such transitions more often, so your agent spends more gradient updates where it can learn most.

> Imagine you are revising with **flash‑cards**. You naturally spend more time on the cards that keep tripping you up and only occasionally glance at the ones you already know. PER teaches a reinforcement‑learning agent to do exactly that.

In terms of implementation, we mainly need two elements - a priority for every transition and a way to sample transitions according to their priority. The priority is defined as

$$p_i = |\delta_i| + \varepsilon$$

where $$\delta_i$$ is the TD error and $$\varepsilon$$ a small constant that ensures that transitions are visited once in a while even if their TD error is zero, i.e. even *easy* cards stay in circulation. The probabilities are then calculated as

$$
P(i) \;=\; \frac{p_i^\alpha}{\sum_k p_k^\alpha},
$$

where $$\alpha$$ controls how strongly to prioritise high TD error transitions ($$\alpha = 0$$ → uniform).
The need to sample with weights changes the requirements for our replay buffer. The **simplest modification would be to implement a linear search**: We store tuples of transitions and their priorities in a flat array. To pick items with a higher priority more often, we can add up all the priorities to get a total and pick a random number between 0 and that total. We then walk through the list, adding up priorities until the running total exceeds the random numbers. That is your selected item. Numbers with a higher priority are more likely to be picked and this likelihood scales linearly with the priority. Technically speaking, we define a probability distribution over items by normalising their priorities and compute the cumulative distribution function (CDF) which is just the running total of probabilities. This technique is called <a href="https://en.wikipedia.org/wiki/Inverse_transform_sampling">inverse transform sampling</a> and can be used for generating sample numbers at random from any probability distribution given its cumulative distribution function.

In this approach, sampling has $$\mathcal{O}(N)$$ time complexity in the buffer size $$N$$ because we need to iterate over all elements. On the up-side, updating probabilites of a given transition works in constant time since we can directly index the relevant priority. Still, we are impatient and do not like linear time for sampling.

To **cut sampling to logarithmic time** we store the priorities in a complete binary tree where each parent holds the sum of its two children—the *sum‑tree*.  The root therefore stores the sum of all priorities and picking a sample becomes a quick left‑right walk down the tree: To find a transition with a given probability, we pick a random number between 0 and the root node's value.
- We pick the root node's left child if the child's value is larger than the random number.
- Else, we pick the right node and subtract the left node's value from the random number.
- We continue this process until we reach a leaf node. That is your selected item.

<figure>
<img src="{{ site.baseurl }}/assets/img/tictactoe-python/21_binary_sum_tree.png" width="100%" alt="Binary sum tree for PER"/>
<figcaption><strong>Figure 6</strong>: Binary sum‑tree with eight leaves.  The root stores the sum of all priorities (43).  Following a random draw of 27, for instance, we picks the right‑left‑right path and lands on the transition with priority 8.</figcaption>
</figure>

Since we only explore one branch of the tree, the computational complexity is reduced from linear to logarithmic. The price to pay is that updating priorities now also comes at a logarithmic cost because we need to update all parent sums from leaf to root.


**PER purposely breaks the i.i.d. assumption**, i.e. that data points are independent and identically distributed. The i.i.d. assumption underlies most theoretical guarantees in machine learning and ensures that the training data is representative of the test data. If the agent mostly sees transitions with high TD errors, it might focus excessively on rare or exceptional situations and perform less well on more typical ones. In practice this bias is removed with **importance‑sampling (IS) weights**

$$
w_i \;=\; \left( \frac{1}{N \, P(i)} \right)^{\beta},
$$

where $$N$$ is the buffer size and $$ \beta\in[0,1]$$ controls the strength of the correction.  The weight rescales the TD‑error of transition *i* before the gradient step.  When $$ \beta=1$$ the update is unbiased; when $$ \beta=0$$ no correction is applied.

Because the largest weight can explode early in training, the common practise is to divide all weights by $$\max_i w_i$$ so that $$w_i\in(0,1]$$.  Most implementations **anneal** $$ \beta$$ linearly from a small value (e.g. 0.4) to 1 over the course of training.


> Early on the agent is like a student cramming the most difficult flashcards; later it wants a balanced rehearsal before the final exam.


#### Spoilt for choice: More hyperparameters
The three important PER hyperparameters are

| symbol | role | typical range |
|---|---|---|
| $$ \alpha$$ | strength of prioritisation | 0.4 – 0.7 |
| $$ \beta_0$$ | initial IS correction | 0.3 – 0.5 |
| $$ \beta_{\text{steps}}$$ | updates until $$ \beta=1$$ | few × 10<sup>3</sup> – 10<sup>5</sup> |

The interactive sweep below shows how they matter in the Tic‑Tac‑Toe experiment:

<!-- Elegant Sweep Variable Selector -->
<div class="sweep-selector">
  <select id="plotSelector" onchange="updateImage()">
    <option value="prb_alpha">Error prioritisation: Alpha</option>
    <option value="prb_beta0">Initial bias correction: Beta</option>
    <option value="prb_beta_steps"># Gradient updates until beta is 1</option>
  </select>
</div>


<!-- Display the selected plot -->
<figure>
<img id="plotImage" src="{{ site.baseurl }}/assets/img/tictactoe-python/per_dqn_prb_alpha_sweep.png" width="100%" alt="Parameter sweep Plot"/>
  <figcaption>
  <strong>Figure 7</strong>: Performance and training loss of PER agent against random minmax agent as a function of PER hyperparameters. Hyperparameters as in Figure 1 with the newly added error priorisation of 0.6, an initial bias correction of 0.4 and 3000, training steps until beta is annealed.
</figcaption>
</figure>


<script>
  function updateImage() {
    const variable = document.getElementById("plotSelector").value;
    const imagePath = `{{ site.baseurl }}/assets/img/tictactoe-python/per_dqn_${variable}_sweep.png`;
    const img = document.getElementById("plotImage");
    img.src = imagePath;
    img.alt = `${variable} Sweep Plot`;
  }
</script>


On Tic‑Tac‑Toe both vanilla DQN and PER‑DQN quickly discover a near‑draw policy, leaving few hard situations to learn from.  To make the difference visible we therefore froze a dataset of **10 000 random‑play transitions** and trained both agents *offline*.  The tables in Figure 8 show how often each board state is replayed.

<figure>
  <iframe src="{{ site.baseurl }}/assets/img/tictactoe-python/22_vanilla_dqn_visit_table.html" width="100%" height="600" frameborder="0"></iframe>
  <iframe src="{{ site.baseurl }}/assets/img/tictactoe-python/23_per_dqn_visit_table.html" width="100%" height="600" frameborder="0"></iframe>
  <figcaption><strong>Figure 8</strong>: Number of state visits per state for vanilla DQN (top) and PER DQN (bottom) during offline training on static set of 10,000 transitions of two random agents playing against one another. PER quickly down‑weights the trivial opening position (state 0) and spends more time on deeper, more decisive boards.Training for 1000 training steps with 2 gradient updates per step and a batch size of 128 on a static training set let to a total of 256,000 state visits. Figure shows the subset of states shown in Figure 1.</figcaption>
</figure>

The corresponding TD‑error histograms in Figure 4 confirm the intuition: by replaying high‑error transitions more often PER flattens the right tail of the error distribution and equalises the visitation count across bins.
<figure>
  <iframe src="{{ site.baseurl }}/assets/img/tictactoe-python/24_td_error_distributions.html" width="100%" height="620" frameborder="0"></iframe>
  <figcaption><strong>Figure 9</strong>: TD error distributions for vanilla DQN and PER DQN during offline training on static set of 10,000 transitions of two random agents playing against one another. Training for 1000 training steps with 2 gradient updates per step and a batch size of 128. Darker colours indicate that states in the bin were visited more frequently.</figcaption>
</figure>

Since PER visits states with high TD errors more often, there are fewer states with high TD errors - the distribution is less extended to the right. Also, the number of state visits per TD error quantile is distributed much more evenly for the PER algorithm.


## Putting it all together
Double DQN, Dueling networks and Prioritized replay complement each other neatly.  In **fewer than fifty lines of code** you can

1. compute the TD‑target with Double DQN,
2. swap the value‑function head for a Dueling head, and
3. replace the FIFO buffer with the PER sum‑tree.

On large, noisy tasks this trio is usually both **faster to learn** and **better at convergence** than plain DQN.  On tiny tabular games such as Tic‑Tac‑Toe the gains are minimal—linear programming or even exhaustive search still win.

In the next post we will take the upgraded agent to a tougher arena: *Bomberman*.  From there the road leads toward the full **Rainbow** combination with multi‑step returns, noisy nets and distributional value functions.
