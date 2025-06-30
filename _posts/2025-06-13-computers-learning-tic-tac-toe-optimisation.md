---
layout: post
title:  "Computers learning Tic-Tac-Toe Pt. 3: Optimisation"
date:   2025-06-13
description: Performance and hyperparameter optimisation for the vanilla DQN algorithm for Tic-Tac-Toe
---

<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

<p class="intro"><span class="dropcap">I</span>n the <a href="https://kunkelalexander.github.io/blog/computers-learning-tic-tac-toe-deep-q/">previous post</a> we built a vanilla Deep Q-Network (DQN) agent for Tic-Tac-Toe and compared it to its tabular cousin.
Today, we look at its hyperparameter optimisation as I find that as all the seemingly arbitrary hyperparameter choices in machine learning algorithms are a fascinating but also unsettling topic. We first study a <a href="https://www.brendangregg.com/flamegraphs.html">flame graph</a> of a DQN training run and then use the <a href="https://optuna.org/">Optuna</a> hyperparameter optimisation framework with the vanilla DQN model.</p>


## Performance optimisation
In the following, we study whether the performance of our vanilla DQN algorithm can be improved. Before we start, it is important that highlight that you should not waste your time or sacrifice the readability of your code for optimisations that turn out to be worthless. In the words of <a href="https://web.archive.org/web/20130731202547/http://pplab.snu.ac.kr/courses/adv_pl05/papers/p261-knuth.pdf">Donald Knuth</a>:
> We should forget about small efficiencies, say about 97% of the time: premature optimization is the root of all evil. Yet we should not pass up our opportunities in that critical 3%.

<a href="https://www.brendangregg.com/flamegraphs.html">Flame graphs</a> are a visualisation of the stack trace of a profiled software created by Brendan Gregg. They allow you to identify performance bottlenecks in your code visually with minimal effort. And flame graphs help to identify the 3% of opportunities Knuth talked about! Let us take a look at a flame graph of my DQN learning routine created with the wonderful
    <a href="https://github.com/benfred/py-spy">py-spy</a>. It is great for interactive vector graphics. For a conventient local analysis, I also recommend
    <a href="https://jiffyclub.github.io/snakeviz/">snakeviz</a>.


<figure>
<object data="{{ site.baseurl }}/assets/img/tictactoe-python/12_profiling.svg"
        type="image/svg+xml"
        width="100%" height="600px">
  Your browser does not support SVGs. You can view it <a href="{{ site.baseurl }}/assets/img/tictactoe-python/12_profiling.svg">here</a>.
</object>
  <figcaption>
      Figure 1: Flame graph for training of vanilla DQN agent. The x-axis represents the stack frame population, while the y-axis shows the call depth. Each rectangle is a stack frame; its width corresponds to the time spent in that function.
    </figcaption>
</figure>


In this run, the `run_training` function dominates the execution time. Most of it is consumed by the `train` function of the vanilla DQN, which itself relies heavily on TensorFlow’s `predict_on_batch` and `train_on_batch`. Since these are well-optimized internals, any inefficiency likely stems from misuse (e.g. improper input formatting or inefficient batch sizes).

Another chunk of time is spent in the `minibatch_to_arrays` function, which converts Python lists to TensorFlow arrays. While this might be optimisable, it only accounts for ~30% of execution time — not worth pursuing unless necessary.

Similarly, the `act` function takes time due to TensorFlow inference calls, which again is expected. One worthwhile optimisation I did make: disabling TensorBoard, which drastically cut training time from 10 to 2 minutes.


## Hyperparameter optimisation


Hyperparameter tuning is the less glamorous sibling of model design, often feeling like adjusting dials in the dark hoping for gold. With parameters like learning rate, discount factor, batch size, and exploration decay all in play, brute-force grid search quickly becomes infeasible.

Still, grid search can be illuminating — especially for **sensitivity analysis** around a known good configuration. The plot below shows the result of sweeping various hyperparameters individually, with all other parameters held constant from the <a href="https://kunkelalexander.github.io/blog/computers-learning-tic-tac-toe-deep-q/">previous post</a>.


<!-- Elegant Sweep Variable Selector -->
<div class="sweep-selector">
  <select id="plotSelector" onchange="updateImage()">
    <option value="hidden_layer">Hidden Layer Size</option>
    <option value="learning_rate">Learning Rate</option>
    <option value="grad_steps">Grad Steps</option>
    <option value="discount">Discount</option>
    <option value="exploration_decay">Exploration Decay</option>
    <option value="batch_size">Batch Size</option>
  </select>
</div>

<!-- Display the selected plot -->
<figure>
<img id="plotImage" src="{{ site.baseurl }}/assets/img/tictactoe-python/dqn_hidden_layer_sweep.png" width="100%" alt="Parameter sweep Plot"/>
  <figcaption>
  Figure 2: Performance and training loss of vanilla DQN agent against random minmax agent as a function of different hyperparameters. The training loss quantifies how well the network fulfills the Bellmann equation. The parameter baseline is set as follows: 3000 training episodes, evaluation every 100 episodes across 100 games, a discount factor of 0.8, learning rate of 0.01 without decay, and initial exploration rate of 1.0 with exponential decay of 0.01 per game down to 0.0. The agent uses a batch size of 128, a replay buffer of size 10,000 with a minimum of 1,000 experiences before training, and two gradient updates per training step. The agents only take legal actions. Shaded areas show standard deviation of draw rate across ten runs with different random seeds. We use a single hidden layer.
</figcaption>
</figure>

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

<script>
  function updateImage() {
    const variable = document.getElementById("plotSelector").value;
    const imagePath = `{{ site.baseurl }}/assets/img/tictactoe-python/dqn_${variable}_sweep.png`;
    const img = document.getElementById("plotImage");
    img.src = imagePath;
    img.alt = `${variable} Sweep Plot`;
  }
</script>

We observe several things:

- **Hidden Layer Size**: At least 64 neurons are needed for good performance. Larger networks reduce training loss due to more degrees of freedom but risk overfitting.
- **Learning Rate**: Even with Adam optimiser, this hyperparameter is critical. Too high, and training diverges. Too low, and it crawls. The best-performing configuration also yields the lowest loss.
- **Gradient Steps**: The number of updates per training step has a limited effect. Fewer updates slow convergence slightly, but the final performance remains largely unchanged.
- **Discount Factor**: Affects convergence speed but not end performance — at least in this setup. It is correlated with the learning rate, likely due to their joint influence on value estimation.
- **Exploration Decay**: Too low a decay hampers training, as the agent stays random for too long. Otherwise, it has little effect — slower decay could simply require more training.
- **Batch Size**: Has minimal influence on performance but impacts training loss. Larger batches reduce variance in gradient updates and thus loss, but may lead to overfitting in more complex environments.


These results raise an important question: What about **interactions between hyperparameters**? Could a smaller network outperform a larger one with the right discount and learning rate, for instance?

To answer this, I turned to the elegant <a href="https://optuna.org/">Optuna</a> framework. Optuna uses intelligent sampling of the search space and supports early pruning of poor runs — making it a great fit for expensive training pipelines. We can set up a study a by specifying and optimisation goal and the hyperparameters to change. After some experimentation, I chose the average draw and victory rate across five training runs for every trial: DQN vs random, random vs DQN, DQN vs Minimax, Minimax vs DQN and DQN vs DQN for every trial. We saw earlier that the training results also vary quite a bit depending on the random number generators, so averaging across several training runs is sensible to avoid falsely identifying an outlier as good parameter set. Still, also averaging over different random number generator seeds would be even better, but time-consuming. The results of the optimisation that ran for 12 hours looks like this.


<figure>
  <iframe src="{{ site.baseurl }}/assets/img/tictactoe-python/13_optuna_optimisation_history.html" width="100%" height="600" frameborder="0"></iframe>
  <figcaption><strong>Figure 2:</strong> Optimization history across 77 trials. Several parameter combinations consistently achieved average draw rates of around 0.98.</figcaption>
</figure>


Following our earlier results, I focused on three parameters I deemed interesting: the learning rate, the discount and the number of neurons in the hidden layer where we keep one hidden layer. The following plots show the breakdown of the optimisation results by these three variables.

<figure>
  <iframe src="{{ site.baseurl }}/assets/img/tictactoe-python/15_optuna_slice_lr.html" width="100%" height="600" frameborder="0"></iframe>
  <iframe src="{{ site.baseurl }}/assets/img/tictactoe-python/15_optuna_slice_hidden_units.html" width="100%" height="600" frameborder="0"></iframe>
  <iframe src="{{ site.baseurl }}/assets/img/tictactoe-python/15_optuna_slice_discount.html" width="100%" height="600" frameborder="0"></iframe>
  <figcaption><strong>Figure 4:</strong> Slice plot showing the relationship between the learning rate and the resulting objective value.</figcaption>
</figure>

The plot suggests some trends that we can verify with a linear regression:
$$
\text{objective} = \beta_0 + \beta_1 \cdot \text{learning rate} + \beta_2 \cdot \text{discount} + \beta_3 \cdot \text{hidden units} + \varepsilon,
$$
where $$\text{objective}$$ is the average draw rate across evaluation games, $$\beta_0$$ is the intercept (baseline performance) and $$\varepsilon$$ is the unexplained error (residuals).

The table below summarizes the model coefficients and global fit statistics:

| Term/Metric            | Value             | p-value       | Interpretation |
|------------------------|-------------------|---------------|----------------|
| **Intercept**          | 0.56 ± 0.08       | < 0.1%        | Baseline performance when all params are 0.|
| **Learning Rate**      | 9.8 ± 3.5         | 0.7%          | Increasing the learning rate leads to a strong improvement of the result — as long as it’s not too high and unstable |
| **Discount Factor**    | 0.18 ± 0.01       | 7.5%          | Moderate effect. Not statistically significant at 5% level, but borderline. |
| **Hidden Units**       | 0.043 ± 0.011     | < 0.1%        | More hidden units improve the results. |
| **R-squared**          | 0.275             | —             | 27.5% of the variation in draw rate explained by the model. |
| **Adjusted R-squared** | 0.245             | —             | Adjusts for the number of predictors. |
| **F-statistic**        | 9.348             | < 0.1%        | Tests if at least one predictor has a non-zero effect. p-value is low - the model is statistically significant |

Finally, we can also try to understand more non-linear effects with a parallel coordinate plot, for instance. You can select parameter ranges like the one for 32 hidden units, for instance, and it becomes obvious that 32 hidden neurons give bad results regardless of the learning rate and discount.
<figure>
  <iframe src="{{ site.baseurl }}/assets/img/tictactoe-python/16_optuna_parallel_coordinate.html" width="100%" height="600" frameborder="0"></iframe>
  <figcaption><strong>Figure 5:</strong> Parallel coordinate plot showing how combinations of hyperparameters affect performance. This visualization is ideal for spotting interactions between parameters.</figcaption>
</figure>


## Conclusion

In this post, we explored techniques for optimizing both the performance and hyperparameters of a vanilla DQN agent. For performance tuning, I strongly recommend using flame graphs — they’re an excellent way to spot bottlenecks and inefficiencies in your implementation.

When it comes to hyperparameter optimization, nothing beats a good initial guess. Simple parameter sweeps are an effective way to check whether you're operating near a "sweet spot." While frameworks like Optuna are powerful and can help develop intuition about parameter sensitivity and interactions, they can also be computationally expensive.

In any case, it is imperative to limit the number of tunable parameters to fewer than 10 — ideally less than four or five — to avoid the curse of dimensionality.

To speed up experimentation, I recommend building a simplified toy model of your problem. You could, for instance, use a smaller, synthetic training set with a fixed number of transitions to control training time and constrain the state space and reduce network size to further accelerate benchmarking.
These simplifications can dramatically reduce training time and help you iterate more quickly on model design and parameter tuning.

In the next post, we’ll move beyond tuning and look at how to improve the DQN algorithm itself.