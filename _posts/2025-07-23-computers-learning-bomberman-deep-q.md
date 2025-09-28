---
layout: post
title:  "Computers learning Bomberman Pt. 2: Deep Q-learning"
date:   2025-09-23
description: Learn how to make computers play Bomberman using the Deep Q-learning algorithm with convolutional neural networks
---

<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

<p class="intro"><span class="dropcap">I</span>n today's post, we use the Deep Q-learning algorithm with convolutional neural networks to learn how to play the Bomberman clone <a href="https://github.com/ukoethe/bomberman_rl">BombeRLe</a> designed for reinforcement learning. You may find the accompanying <a href="https://github.com/KunkelAlexander/bomberman_rl"> Python code on GitHub</a>. For an introduction to Deep Q-learning, please refer to the  <a href="https://kunkelalexander.github.io/blog/computers-learning-tic-tac-toe-deep-q/">Tic-Tac-Toe series</a> on this blog.</p>

<img src="{{ site.baseurl }}/assets/img/bomberle-python/15_bomberle_cnn_architecture.png" width="100%" alt="">

<style>
  figure {
    margin-bottom: 20px;
  }

  figcaption {
    margin-top: 10px;
  }
</style>



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

## How to apply Deep Q-Learning to BombeRLe?

In the <a href="https://kunkelalexander.github.io/blog/computers-learning-bomberman-tabular-q/">last post</a>, we explored BombeRLe’s game mechanics, its reward system, how to design a feature representation for tabular Q-learning, and what the training process and resulting agent performance looked like. This time, we’ll focus on what needs to change to make Deep Q-Learning (DQN) work. Specifically, we need a suitable state representation, a network architecture, and an optimized training process. For this, we’ll be reusing the DQN implementation from <a href="https://kunkelalexander.github.io/blog/computers-learning-tic-tac-toe-advanced-deep-q/">here</a>, which already includes Prioritized Experience Replay, Double DQN, and Dueling DQN. Let’s start by looking at the state representation and the network architecture.

In Tic-Tac-Toe, Deep Q-learning was straightforward: I simply fed the $$9$$ input fields into a fully connected layer of a neural network, without worrying about feature design. We could take a similar approach in BombeRLe. The board is larger - $$9\times9=81$$ fields in the small version I’ll keep using—but that’s the only real difference. We could label each input state (empty, wall, crate, player, coin, bomb, etc.) and feed those directly into the network. Figure 1 illustrates such an encoding. However, it will probably lead to suboptimal performance as the neural network would need to learn to interpret categorical cell types from continuous numerical inputs.


<figure>
  <img src="{{ site.baseurl }}/assets/img/bomberle-python/13_cnn_simple_representation.png"  width="100%" alt="">
  <figcaption>Figure 1: Simple BombRLe state representation for Deep Q-Learning. The game board is encoded as a 9×9 array, where each cell takes on a discrete value (wall, crate, coin, agent, etc.). While easy to implement, this representation is problematic in practice: a neural network would need to learn to interpret categorical cell types from continuous numerical inputs, which is inefficient and often leads to poor generalization.</figcaption>
</figure>

A more effective approach is *one-hot encoding*. Instead of compressing all cell types into a single grid of integers, we give each type its own binary layer. For example, one $$9\times9$$ array marks where the walls are (1 = wall, 0 = no wall), another marks crates, another marks coins, and so on. Stacking these layers creates a set of “feature maps” that together describe the state of the game. This format is far easier for a neural network to interpret, since it only needs to detect patterns of 0s and 1s within each layer, rather than decode arbitrary categories. Figure 2 illustrates the idea with the encoding that I chose.


<!-- Elegant Input-Layer Selector -->
<div class="sweep-selector">
  <select id="inputLayerSelector" onchange="updateInputLayer()">
    <option value="input_00_Wall.png">Wall</option>
    <option value="input_01_Free.png">Free</option>
    <option value="input_02_Crate.png">Crate</option>
    <option value="input_03_Bomb.png">Bomb</option>
    <option value="input_04_Danger.png">Danger</option>
    <option value="input_05_Explosion.png">Explosion</option>
    <option value="input_06_Coin.png">Coin</option>
    <option value="input_07_Player.png">Player</option>
    <option value="input_08_Opponent.png">Opponent</option>
    <option value="input_09_PlayerCanDrop.png">Player can drop bomb?</option>
    <option value="input_10_OpponentCanDrop.png">Opponent can drop bomb?</option>
  </select>
</div>

<!-- Display the selected input-layer image -->
<figure>
  <!-- Change 00014 to the frame you want to show; keep folder structure in sync with your exports -->
  <img id="inputLayerImage"
       src="{{ site.baseurl }}/assets/img/bomberle-python/14_cnn_feature_channels/00014/input/input_00_Wall.png"
       width="100%"
       alt="Input layer: Wall"/>
  <figcaption>
    Figure 2: BombRLe one-hot encoding for Deep Q-Learning. The game board is represented as a stack of eleven 9×9 binary arrays, each marking the positions of walls, crates, coins, agents, etc. Special layers like “bomb,” “danger,” and “explosion” hold values between 0 and 1 to encode bomb timers, blast danger, and explosion durations. Higher values indicate higher risk. The danger layer, for example, encodes the smallest bomb timer that threatens a given tile. Use the dropdown to view each of the 11 channels overlaid on the game board.
  </figcaption>
</figure>

<script>
  // Base path to the exported PNGs for a single frame (e.g., 00014).
  // If you change frame, just change frameId or build a second selector if you like.
  const frameId = "00014";
  const basePath = `{{ site.baseurl }}/assets/img/bomberle-python/14_cnn_feature_channels/${frameId}/input/`;

  const files = [
    {file: "input_00_Wall.png",             label: "Wall"},
    {file: "input_01_Free.png",             label: "Free"},
    {file: "input_02_Crate.png",            label: "Crate"},
    {file: "input_03_Bomb.png",             label: "Bomb"},
    {file: "input_04_Danger.png",           label: "Danger"},
    {file: "input_05_Explosion.png",        label: "Explosion"},
    {file: "input_06_Coin.png",             label: "Coin"},
    {file: "input_07_Player.png",           label: "Player"},
    {file: "input_08_Opponent.png",         label: "Opponent"},
    {file: "input_09_PlayerCanDrop.png",    label: "Player can drop bomb?"},
    {file: "input_10_OpponentCanDrop.png",  label: "Opponent can drop bomb?"}
  ];

  // Optional: preload to reduce flicker
  (function preload() {
    files.forEach(({file}) => {
      const img = new Image();
      img.src = basePath + file;
    });
  })();

  function updateInputLayer() {
    const select = document.getElementById("inputLayerSelector");
    const chosen = select.value;
    const chosenLabel = select.options[select.selectedIndex].text;
    const img = document.getElementById("inputLayerImage");
    img.src = basePath + chosen;
    img.alt = `Input layer: ${chosenLabel}`;
  }
</script>


Note that this encoding is not centered on the player. For scaling to larger boards, it often makes sense to center the input around the player and fix the field of view to a certain size (e.g. still 9×9). With this encoding, we could feed the board into fully connected layers, just like in Tic-Tac-Toe. But in practice, this approach is inefficient: the network needs to learn spatial patterns (like corridors, dead-ends, or safe zones) from scratch, and dense layers are not well suited for spatial data. Instead, convolutional layers are a better fit.

You can think of convolutional layers as a collection of small “pattern detectors” that slide across the grid. At first, they recognize local structures such as walls, bombs, or coins in a small neighborhood. When stacked, these detectors combine into larger and more abstract features, such as escape routes or dangerous areas. By the time the output reaches the fully connected layers, the network has already distilled the raw input into (hopefully) meaningful high-level features.

Figure 3 illustrates the neural network architecture I use in the following. The model is built from a stack of convolutional layers, followed by a pooling layer and a duelling head with fully connected layers. The pooling layer averages the spatial output of the last convolutional layer, reducing it to 64 values that serve as high-level features.

I also tried flattening the output of the third convolution directly into a linear array with 64 × 9 × 9 elements. Feeding this into the fully connected layer did work, but it resulted in a layer with over ~600k parameters. The network then showed poor generalisation: it played well against other agents but completely failed in unseen environments such as the coin-heaven scenario (with only coins and no enemies or crates).

Introducing the pooling layer fixed this issue by both improving generalisation and balancing the number of parameters between the convolutional and fully connected layers. Reducing the number of filters in the convolutional layers led to worse performance, so I consider 64 filters the minimum required for this task.

<figure>
  <img src="{{ site.baseurl }}/assets/img/bomberle-python/15_bomberle_cnn_architecture.png"  width="100%" alt="">
  <figcaption>Figure 3: BombeRLe convolutional neural network with duelling head. The input state is represented as a 9 × 9 × 11 tensor, which is processed by three convolutional layers with 64 filters each. I use a stride of 1 to capture every input pixel, a dilation of 1 for simplicity, and 3 × 3 kernels. This results in approximately 6k, 18k, and 37k parameters in the first, second, and third convolutional layers, respectively.

The output of the third convolutional layer is spatially averaged, leaving only the 64 channels as high-level features. These are passed through a fully connected layer with ~8k parameters (64 × 128 + 128), followed by the duelling head, which consists of two fully connected layers with ~16k parameters.

In total, the network has roughly 100k parameters, with about 60% in the convolutional layers and 40% in the fully connected layers.</figcaption>
</figure>


### The training process

Most hyperparameters & rewards are unchanged compared to the <a href="https://kunkelalexander.github.io/blog/computers-learning-bomberman-tabular-q/">last post</a>. For reference, the unchanged parameters are:
- **Coins**: 1 point per coin
- **Kills**: 5 points per enemy killed
- **Crate density**: Default is `0.75`, can be adjusted by scenario
- **Number of coins**: Default is `9`, can be adjusted by scenario
- **Board size**: My default is `9x9` (including walls)
- **Bombs**: Explode in the `4`th frame after being dropped, explosion affects a radius of `3` fields and is present for `2` frames
- **Discount factor (γ)**: 0.8
- **Max steps**: 400

I also left the reward system unchanged:

| Event              | Reward  | Description                              |
|--------------------|---------|------------------------------------------|
| COIN_COLLECTED     | +0.20   | Agent collected a coin                   |
| KILLED_OPPONENT    | +1.00   | Agent killed another agent               |
| CRATE_DESTROYED    | +0.10   | Agent destroyed a crate                  |
| BOMB_DROPPED       | +0.02   | Agent dropped a bomb                     |
| KILLED_SELF        | −0.90   | Agent killed itself                      |
| GOT_KILLED         | −1.00   | Agent got killed by opponent             |
| WAITED             | −0.02   | Agent performed a WAIT action            |
| INVALID_ACTION     | −0.02   | Agent performed an invalid action        |

Note that the value of *KILLED_SELF* should ideally be slightly positive. In the case of suicide, the agent receives both the *GOT_KILLED* and *KILLED_SELF* rewards. Since suicide is preferable to being killed by an opponent—because no one else scores points—it makes sense to reflect that in the reward structure. However, simplifying the reward scheme actually degraded performance. I found it important to strongly incentivise the agent to drop bombs and destroy crates.

I used the following hyperparameters for DQN:

* **Learning rate (α):** 3e-4. Higher rates led to instability, while lower ones slowed training.
* **Batch size:** 64.
* **Gradient steps:** 4. This means four gradient updates per training update. With up to 400 transitions per game episode, we insert 400 transitions into the buffer but only train on 4 × 64 = 256 samples. As a result, each transition is visited at most once, making the replay buffer almost ineffective. I plan to address this in a follow-up post.
* **Exploration:** ε decays from 1 to 0.1 with a per-episode decay factor of 0.999. After ~1,000 episodes, ε is around 0.3, and it reaches 0.1 after ~3,000 episodes.
* **Prioritised experience replay:** Training starts once the buffer contains 10,000 transitions (roughly 1,000 episodes). The buffer capacity is 100,000 transitions (about 250 full episodes). I anneal toward unbiased updates over 100,000 gradient steps (~25,000 episodes).
* **Double DQN:** The target network is updated with a hard copy of the online network every 10 gradient updates. Larger intervals, such as 30, also worked well in my experiments.


## Bombs are all that counts

I simulate a full game in the *classic* scenario with multiple agents:

* A **rule-based agent**
* A **rule-based agent** based on the simplified state representation from the tabular Q-learning post (*the representator*)
* A **peaceful agent** that moves randomly and never places bombs
* A **Deep Q-learning agent** (*cnn_allstar*)

Training was run for 100,000 episodes. On my laptop CPU, this translated to about 2–4 episodes per second, meaning several hours of runtime. I suspect there are still performance bottlenecks in the code, but a quick flame graph analysis didn’t reveal anything obvious. A large fraction of computation time is spent in convolutional and fully connected layer updates, specifically matrix multiplications.

Running on a GPU did not noticeably accelerate training, likely because communication overhead dominates execution time for a network of this size (~0.5 MB). After 100,000 training episodes, the DQN agent (*cnn_allstar*) plays reasonably well, often achieving higher scores than both the rule-based agent and the representator.


<figure>
  <img src="{{ site.baseurl }}/assets/img/bomberle-python/16_cnn_allstar.gif"  width="100%" alt="">
  <figcaption>Figure 4: CNN allstar (pink) playing five episodes against the rule-based agent (yellow), the representator (blue) and the tabular-Q allstar (green) after 100,000 rounds of training.</figcaption>
</figure>

Next, we assess the performance of the CNN allstar by studying the averaged results of 1,000 games between the CNN allstar agent, a rule-based agent, the tabular-Q allstar agent from the previous post and the representator.

| Category | CNN allstar | Rule-based agent | Tabular-Q allstar | Representator |
|----------|-----------------|----------------|----------|----------------|
| bombs | 35 | 6.9 | 11 | 13 |
| coins | 2.7 | 1.5 | 2.6 | 2.2 |
| crates | 6.1 | 4.6 | 4.4 | 5.8 |
| invalid | 1.9 | 4.7 | 1.6 | 3.4 |
| kills | 0.55 | 0.17 | 0.21 | 0.28 |
| moves | 2.1e+02 | 90 | 1.2e+02 | 1.3e+02 |
| score | 5.4 | 2.4 | 3.6 | 3.6 |
| steps | 2.6e+02 | 1.1e+02 | 1.4e+02 | 1.5e+02 |
| suicides | 0.26 | 0.37 | 0.52 | 0.47 |
| time | 0.39 | 0.065 | 0.093 | 0.077 |


Overall, the CNN allstar agent **places more bombs and is a stronger killer than the built-in rule-based agent, the representator, and the tabular Q allstar**. It consistently achieves higher scores than the other agents while committing fewer suicides.

The training process is shown in Figure 5. Without the duelling head—when feeding the one-hot encoding directly into fully connected layers—the DQN agent’s performance plateaued at an average score of about 4. Flattening the convolutional output without pooling allowed the CNN allstar to occasionally reach scores around 8, but training stability was generally poorer.


<figure>
  <img src="{{ site.baseurl }}/assets/img/bomberle-python/17_cnn_allstar_training.png"  width="100%" alt="">
  <figcaption>Figure 5: Average training score of the CNN allstar during 100,000 episodes of training. The CNN allstar consistenly beats the other agents after around 50,000 episodes of training.</figcaption>
</figure>


## Extrapolate to the unseen?
One important capability the agent should have, at least in theory, is the ability to extrapolate to unseen situations.

The figure below demonstrates that the CNN allstar, despite never encountering the coin-heaven scenario during training, is sometimes able to collect all coins. This demonstrates that the network architecture does not terribly overfit the training data.

<figure>
  <img src="{{ site.baseurl }}/assets/img/bomberle-python/18_cnn_coin_grabber.gif" width="100%" alt="">
  <figcaption>Figure 6: CNN allstar successfully collecting all coins after 100,000 rounds of training against other agents. Notably, the agent had never seen the coin-heaven scenario during training, yet it still manages to complete the task in this episode. </figcaption>
</figure>


## What do the convolutional layers actually see?

One thing I’ve been curious about is whether we can actually *see* the high-level features the convolutional layers are pulling out of the one-hot encoded data. As a first step, I plotted the weights of the *cnn_allstar*’s convolutional filters (ignoring biases) after 100,000 episodes of training—shown in Figure 7.

Some of the filters look familiar, with patterns that resemble standard edge detectors. But beyond that, it’s hard to say if the weights are really “reasonable” or interpretable in any intuitive sense.

<figure>
  <img src="{{ site.baseurl }}/assets/img/bomberle-python/19_conv_weights.png" width="100%" alt="">
  <figcaption>Figure 7: Convolutional weights of the *cnn_allstar* (see also Figure 3) after 100,000 training episodes. The network has three convolutional layers, each with 64 filters of size 3×3. The figure shows all 3 × 64 = 192 convolution heads as 3×3 bitmaps. Rows 1–4 (16 columns each) correspond to the first convolutional layer, rows 5–8 to the second, and rows 9–12 to the third. </figcaption>
</figure>

To get something more interpretable, we can instead look at **activations**. For example, the first convolutional layer maps the 9 × 9 × 11 input into a 9 × 9 × 64 output. Plotting these 64 activation maps side by side—and overlaying them with the original game state—gives us a sense of what each filter responds to. Figure 8 shows these activation patterns across the layers.

<div class="sweep-selector" style="display:flex; gap:0.75rem; flex-wrap:wrap; align-items:end;">
  <div style="min-width:300px;">
    <label for="idxSlider">Frame <span id="idxValue">#14</span></label>
    <input id="idxSlider" type="range" min="0" max="14" step="1" value="14" style="width:100%;" list="idxTicks">
    <datalist id="idxTicks">
      <option value="0"></option><option value="2"></option><option value="4"></option>
      <option value="6"></option><option value="8"></option><option value="10"></option>
      <option value="12"></option><option value="14"></option>
    </datalist>
    <div style="display:flex; justify-content:space-between; font-size:0.85em; opacity:0.7;">
      <span>Frame 0</span><span>Frame 14</span>
    </div>
  </div>

  <label>
    Layer
    <select id="layerSelect"></select>
  </label>

  <div style="min-width:300px;">
    <label for="filterSlider">Filter <span id="filterValue">Conv1 · f00</span></label>
    <input id="filterSlider" type="range" min="0" max="63" step="1" value="0" style="width:100%;" list="filterTicks">
    <datalist id="filterTicks">
      <option value="0"></option><option value="8"></option><option value="16"></option>
      <option value="24"></option><option value="32"></option><option value="40"></option>
      <option value="48"></option><option value="56"></option><option value="63"></option>
    </datalist>
    <div style="display:flex; justify-content:space-between; font-size:0.85em; opacity:0.7;">
      <span>f00</span><span id="filterMaxLabel">f63</span>
    </div>
  </div>
</div>


<figure>
  <img id="convImage" src="" width="100%" alt="Conv activation overlay"/>
  <figcaption>
    Figure 8: Convolutional layer activations overlaid on the game screenshot.
    Use the sliders and dropdown to explore different indices, layers, and filters.
  </figcaption>
</figure>

<script>
  // Base path for exported PNGs
  const BASE_PATH = `{{ site.baseurl }}/assets/img/bomberle-python/14_cnn_feature_channels`;

  // Map numeric layer (1..3) → folder + filename base + (optional) max filters
  // 🔧 Edit to match your export folders & file bases.
  const LAYERS = {
    1: { folder: "conv_conv2d_3",   fileBase: "conv2d_3",   maxFilters: 64 },
    2: { folder: "conv_conv2d_4", fileBase: "conv2d_4", maxFilters: 64 },
    3: { folder: "conv_conv2d_5", fileBase: "conv2d_5", maxFilters: 64 }
  };

  // Defaults
  const DEFAULTS = { idx: 14, layer: 1, filter: 0 };

  // Elements
  const idxSlider      = document.getElementById("idxSlider");
  const idxValue       = document.getElementById("idxValue");
  const layerSelect    = document.getElementById("layerSelect");
  const filterSlider   = document.getElementById("filterSlider");
  const filterValue    = document.getElementById("filterValue");
  const filterMaxLabel = document.getElementById("filterMaxLabel");
  const imgEl          = document.getElementById("convImage");

  // Helpers
  function zpad(num, width) {
    const s = String(num);
    return s.length >= width ? s : ("0".repeat(width - s.length) + s);
  }
  function idxString(n) { return zpad(n, 5); }     // 0 → "00000"
  function fString(n)   { return "f" + zpad(n, 2); } // 7 → "f07"

  function buildSrc(idxNum, layerNum, filterIdx) {
    const idx = idxString(idxNum);
    const layer = LAYERS[layerNum];
    return `${BASE_PATH}/${idx}/${layer.folder}/${layer.fileBase}_${fString(filterIdx)}.png`;
  }

  function updateFilterUI(maxF) {
    const maxIdx = Math.max(0, (maxF ?? 64) - 1);
    filterSlider.max = maxIdx;
    if (parseInt(filterSlider.value, 10) > maxIdx) {
      filterSlider.value = String(maxIdx);
    }
    filterMaxLabel.textContent = fString(maxIdx);
    filterValue.textContent = fString(parseInt(filterSlider.value, 10));
  }

  function updateIdxUI() {
    const idxNum = parseInt(idxSlider.value, 10);
    idxValue.textContent = idxString(idxNum);
  }

  function updateImage() {
    const idxNum = parseInt(idxSlider.value, 10);
    const layer  = parseInt(layerSelect.value, 10);
    const f      = parseInt(filterSlider.value, 10);
    const src    = buildSrc(idxNum, layer, f);
    imgEl.src    = src;
    imgEl.alt    = `Index ${idxString(idxNum)} · Layer ${layer} · ${fString(f)}`;
  }

  function preloadNeighbors() {
    const idxNum = parseInt(idxSlider.value, 10);
    const layer  = parseInt(layerSelect.value, 10);
    const f      = parseInt(filterSlider.value, 10);

    const idxNeighbors = [idxNum - 1, idxNum + 1].filter(n => n >= parseInt(idxSlider.min,10) && n <= parseInt(idxSlider.max,10));
    const fNeighbors   = [f - 1, f + 1].filter(n => n >= parseInt(filterSlider.min,10) && n <= parseInt(filterSlider.max,10));

    // Preload next/prev filter for current index
    fNeighbors.forEach(ff => {
      const i = new Image();
      i.src = buildSrc(idxNum, layer, ff);
    });
    // Preload current filter for adjacent indices
    idxNeighbors.forEach(ii => {
      const i = new Image();
      i.src = buildSrc(ii, layer, f);
    });
  }

  function populate() {
    // Layers
    Object.keys(LAYERS).forEach(k => {
      const opt = document.createElement("option");
      opt.value = k;
      const info = LAYERS[k];
      opt.textContent = `Layer ${k} (${info.fileBase})`;
      layerSelect.appendChild(opt);
    });

    // Defaults
    idxSlider.value   = String(DEFAULTS.idx);
    layerSelect.value = String(DEFAULTS.layer);
    filterSlider.value = String(DEFAULTS.filter);

    updateIdxUI();
    updateFilterUI(LAYERS[DEFAULTS.layer].maxFilters);
    updateImage();
    preloadNeighbors();
  }

  // Events
  idxSlider.addEventListener("input", () => { updateIdxUI(); updateImage(); });
  idxSlider.addEventListener("change", preloadNeighbors);

  filterSlider.addEventListener("input", () => {
    filterValue.textContent = fString(parseInt(filterSlider.value, 10));
    updateImage();
  });
  filterSlider.addEventListener("change", preloadNeighbors);

  layerSelect.addEventListener("change", () => {
    const layer = parseInt(layerSelect.value, 10);
    updateFilterUI(LAYERS[layer].maxFilters);
    updateImage();
    preloadNeighbors();
  });

  document.addEventListener("DOMContentLoaded", populate);
</script>


## Conclusion

In this post, we trained a DQN algorithm with a convolutional neural network to play a Bomberman clone. The approach proved effective: the final agent outperformed the tabular Q-agent developed earlier, and it did so without heavy reliance on handcrafted features. Training this agent was more complex than training the tabular Q-agent, but in my view the solution is far more elegant.

Looking ahead, I’d like to refine the state encoding—for example, by stacking multiple frames to provide temporal context—and also explore how well the agent can learn directly from raw RGB image input.