---
layout: post
title:  "Computers learning Bomberman Pt. 1: Tabular Q-learning"
date:   2025-07-23
description: Learn how to make computers play Bomberman using the Q-learning algorithm
---

<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

<p class="intro"><span class="dropcap">I</span>n today's post, we use the tabular Q-learning algorithm to learn how to play the Bomberman clone <a href="https://github.com/ukoethe/bomberman_rl">BombeRLe</a> designed for reinforcement learning. You may find the accompanying <a href="https://github.com/KunkelAlexander/bomberman_rl"> Python code on GitHub</a>. For an introduction to tabular Q-learning, please refer to the  <a href="https://kunkelalexander.github.io/blog/tic-tac-toe/">Tic-Tac-Toe series</a>.</p>

<img src="{{ site.baseurl }}/assets/img/bomberle-python/1_gameplay.gif" width="100%" alt="">


## BombeRLe: Game Mechanics Explained

In BombeRLe, up to four agents compete by collecting coins and placing bombs to destroy crates and eliminate enemies. The goal of the game is to collect as many coins and kill as many enemies as possible.
- The game is turn-based and each agent has **6 possible actions**: Move up, right, down, left, place a bomb and wait
- Agents **act in random order**. This means that depending on who moves first walking into another player moving away will sometimes result in waiting, sometimes in moving.
- Walking into an obstacle (wall, bomb, crate) results in waiting instead but an agent can stand on its own bomb after placing it.
- Agents **blow up crates using bombs** and **collect coins by walking through them**.
- Every player has one life and **dies immediately** when stepping into an explosion. If an agent kills another player and dies at the same time, this counts as kill.
- Players can place **only one bomb at a time**. The timeline of a bomb is detailed in the table below and visualised in Figure 1.


| Frame | State             | Description                                      |
|-------|-------------------|--------------------------------------------------|
| 1     | Agent drops the bomb  | New bomb with time to explosion `t = 4`|
| 2     | Bomb ticking      | Bomb timer counts down to `t = 3`              |
| 3     | Bomb ticking      | Bomb timer counts down to `t = 2`              |
| 4     | Bomb ticking      | Bomb timer counts down to `t = 1`              |
| 5     | Bomb explodes     | Explosion at `t = 0`, affecting `3` tiles |
| 6     | Explosion remains | Explosion lingers for 1 frame |
| 7+    | Agent can place a new bomb  | Harmless smoke remains for 2 frames |


<figure>
  <img src="{{ site.baseurl }}/assets/img/bomberle-python/2_explain_bombs.gif"  width="100%" alt="">
  <figcaption>Figure 1: Bomb game mechanics with default settings. The agent places a bomb, then moves away, waits for three frames and moves back into the smoke once the explosion has subsided.</figcaption>
</figure>

### Your best friend: parameters
The game allows for a number of settings that can be found in `settings.py`
- **Coins**: 1 point per coin
- **Kills**: 5 points per enemy killed
- **Crate density**: Default is `0.75`, can be adjusted by scenario
- **Number of coins**: Default is `9`, can be adjusted by scenario
- **Board size**: Default is `13x13` (including walls)
- **Bombs**: Explode in the `4`th frame after being dropped, explosion affects a radisu of `3` fields and is present for `2` frames

## A good state representation for tabular Q-learning should be small

Tabular Q-learning requires the game to be represented through a finite number of states. Bomberman is a discrete, turn-based game, hence, perfectly suitable for tabular Q-learning. But you may ask whether we can simply number all possible board states and put them into the table It turns out that this is unfeasible: assuming a small $$9\times9$$ board with $$7\times7−9 = 40$$ free fields that can either be free, occupied by a crate, a coin, an agent, or an explosion gives roughly $$4^{40} = 2^{80}\approx 10^{24}$$ states. And this does not even account for bomb availability, explosion timers, or bomb timers. A Q-table for six possible actions (`UP, RIGHT, DOWN, LEFT, BOMB, WAIT`), using 4-byte floating point numbers, would require around $$10^{10}$$ petabytes. Beyond memory requirements, trainability is another important consideration: a more granular state representation captures more nuances of the game, but requires more training data. Unlike DQN, tabular Q-learning does not extrapolate. So, the agent will necessarily fail in states it has not seen before. Therefore, we need to find a simplified representation of the game state. The upper size limit will be a 32-bit representation, corresponding to a roughly 100 GB-table if all states are populated. So, you should aim for less than 30 bits in most cases. After some experimentation, I settled for the following **18-bit representation**:

| Bits      | Field                            | Description                                                                                  |
|-----------|----------------------------------|----------------------------------------------------------------------------------------------|
| 0–2   | Neighbour (UP)                    | 3-bit code for object in UP direction: EMPTY, WALL, COIN, CRATE, ENEMY, BOMB, DANGER |
| 3–5   | Neighbour (RIGHT)                 | 3-bit code for object in RIGHT direction                                                     |
| 6–8   | Neighbour (DOWN)                  | 3-bit code for object in DOWN direction                                                      |
| 9–11  | Neighbour (LEFT)                  | 3-bit code for object in LEFT direction                                                      |
| 12–13 | Direction to nearest object      | 2-bit code: 00 = UP, 01 = RIGHT, 10 = DOWN, 11 = LEFT                                        |
| 14–15 | Type of nearest object           | 2-bit code: 00 = NONE, 01 = ENEMY, 10 = CRATE, 11 = COIN                                     |
| 16    | Action safe: BOMB                | 1 = safe, 0 = unsafe                                                                         |
| 17    | Action safe: WAIT                | 1 = safe, 0 = unsafe                                                                         |


The four 3-bit fields summarize the nearest tile in each direction:

- WALL, CRATE, BOMB, ENEMY, COIN, EMPTY: as seen now (no future prediction).
- DANGER: stepping onto that tile now makes survival impossible given all current bombs and blast timers.
- one unused state that could be used to distinguish bombs with different timers, enemies with and without bombs etc.

I define an action as "safe" if, taking into account all current bombs and explosions, executing that move still allows the agent to survive through the remaining blasts. That is, there exists at least one follow-up move or WAIT that avoids all danger until all bombs have exploded. This gives the agent a basic capacity to avoid walking into flames, trapping itself with its own bombs (a problem in previous designs), or failing to escape in time. It does *not* protect against being intentionally trapped by another agent or bombs being set off by other bombs, for instance. So this representation will not lead to an ideal agent. But in practice, the trained agent should no longer easily kill itself, it should reliably dodge explosions, break crates, collect coins, and occasionally kill another agent if it gets lucky. It will, however, still lose to a smart opponent that traps it in tunnels, for instance.

These 12 bits allow the agent to perceive its immediate surroundings and assess dangerous situations to some extent. However, the agent still lacks any sense of the wider game board. To remedy this, I introduce **4 more bits that encode the direction and type of the nearest object of interest**. This encoding implies the agent can only pursue a single object of interest at a time. It will not be aware of, say, two coins at equal distance. If the agent is standing adjacent to the object of interest, the directional encoding will be inaccurate, since I chose not to include `WAIT` as a direction. But this is acceptable: the agent "sees" the object and is therefore in a clearly distinct state. In my understanding, this design choice should not negatively affect the performance of the tabular Q-agent.


The code runs a breadth-first search (BFS) to locate the nearest coins, crates, and agents. We guide the agent toward the nearest agent not separated by walls or crates. If there is none, then the nearest walkable coin. Failing that, the nearest walkable crate. If none of those are available, the  should generally be over.

Once the agent knows where it can go, two final choices remain: Should the agent wait or place a bomb? To answer this question, I added **2 more bits indicating whether waiting and bombs are safe or unsafe** using the same logic as for assessing dangerous direction. Waiting is safe if no bombs can kill the agent in its current position; placing a bomb is safe if waiting is safe and the agent has an escape route before the bomb goes off.

This representation consumes **18 bits in total**, translating to $$2^{18} = 262,144$$ possible states. A Q-table with 6 actions and 4-byte floats would then require about $$2^{18} * 6 * 4 \text{ bytes} = 6,291,456 \text{ bytes} \approx 6 \text{ megabytes}$$.

However, since many states are never visited during training, I store the Q-table as a hash table (i.e. a Python dictionary). This trades a slight performance penalty on lookup for a reduction in memory use — down to just a few hundred kilobytes for a typical training run.

The animation shown in Figure 2 illustrates this encoding in practice. It shows the raw bit string, its division into the logical components described above, and the corresponding decoded categories. I found this reverse translation very helpful for debugging the encoding and would strongly recommend building a debug screen when designing state representations.

<figure>
  <img src="{{ site.baseurl }}/assets/img/bomberle-python/3_debug_state.gif"  width="100%" alt="">
  <figcaption>Figure 2: Debug screen for simplified state representation. At every frame, we see the 18 bit representation of the state with a translation back into features. </figcaption>
</figure>


### A good reward system helps even when in-game rewards are scarce

Ideally, our reward system should mirror the games reward system yielding the same relative rewards for collecting coins and killing players. However, in practice rewards are sparse, especially when the player has to blow up crates to find coins and plays against strong opponents. Therefore, I introduce additional rewards that help the agent learn to explore the game.

I mirror the rewards of the game with coin collected and killed opponent. Additionally, the player is rewarded for surviving the round and destroying crates. It gets punished for killing itself and getting killed and there is a slight penalty for inaction. The latter reflects that the enemy will have an easier time killing a waiting target even when the agent itself thinks that waiting is as good as moving.


| Event              | Reward  | Description                              |
|--------------------|---------|------------------------------------------|
| COIN_COLLECTED     | +0.20   | Agent collected a coin                   |
| KILLED_OPPONENT    | +1.00   | Agent killed another agent               |
| CRATE_DESTROYED    | +0.10   | Agent destroyed a crate                  |
| KILLED_SELF        | −1.00   | Agent killed itself                      |
| SURVIVED_ROUND     | +1.00   | Agent survived until the end of the round|
| GOT_KILLED         | −0.10   | Agent got killed by opponent or bomb     |
| WAITED             | −0.02   | Agent performed a WAIT action            |

## Faster Training Through Demonstration

For the Tic-Tac-Toe agent, I simply let the agent play games and updated the Q-table on the fly. This time, I would like to accelerate training to enable a more thorough sweep of the hyperparameter space.

One key opportunity I overlooked in Tic-Tac-Toe was the availability of a near-perfect demonstrator: the Minimax agent. Similarly, in the case of BombeRLe, the developers kindly provide a rule-based agent that performs reasonably well. We can leverage this to speed up training significantly by pre-recording a large number of transitions and using them to pre-train the Q-table before starting real-time learning. This is called **offline learning**. In contrast, sampling training data from the agent while it is learning is called **online learning**.

Offline learning with a demonstrator allows the agent to explore useful game states right from the beginning—states it would otherwise only encounter much later during training. In practice, I record **50,000 transitions** from the demonstrator and use them to update the Q-table before performing any live training or evaluation. For context, a full training run with online updates can take an hour, whereas updating the Q-table with 50k recorded transitions takes **around 60 seconds**. This enables rapid experimentation across different hyperparameter configurations.

The key speed gain comes from the fact that we only need to compute the computationally expensive game state to integer state embedding **once**. These embeddings can then be reused across multiple hyperparameter runs.

## Money, money, money

We start training in a simplified environment without enemies or crates, a small $$9\times9$$ board and coins everywhere.

### Hyperparameters

* **Discount factor (γ)**: 0.8
* **Learning rate (α)**: 1e-2
* **Initial Q-values**: -1

Since we’re using off-policy learning (based on demonstrator trajectories), the agent may not explore all available states. Initializing Q-values to a **high value** would encourage the agent to explore, but also risks overestimating actions not demonstrated. Instead, by initializing Q-values to a **negative number**, the agent is pessimistic by default, and learns to value good actions based on demonstrator input.

### Adaptive Learning Rate

We further improve convergence using an adaptive learning rate $$\alpha = lr_0 / (1 + N_{visits}(state, action)).$$ This formulation decreases the learning rate as the agent gathers more experience with a specific `(state, action)` pair. Early in training, updates are larger and help the agent adapt quickly. Later, as visit counts increase, the learning rate decays, stabilizing the learned Q-values and reducing variance. This strategy helps avoid overfitting to noisy or rare transitions and provides **better convergence guarantees**.

### Performance

<figure>
  <img src="{{ site.baseurl }}/assets/img/bomberle-python/4_coin_grabber.gif"  width="100%" alt="">
  <figcaption>Figure 3: Tabular Q-agent successfully collecting coins. The agent comfortably navigates the map and reliably collects all coins.</figcaption>
</figure>

The agent reliably collects all coins. However, its path is not fully optimized: it always moves toward one of the nearest coins but doesn't plan multi-step routes that could reduce total steps. This is due to the BFS algorithm, which only targets the next closest coin and ignores multi-coin path efficiency.

Additionally, the agent’s behavior reflects a **design choice**: the pathfinding algorithm always prefers the closest object of interest following the fixed order: **Up → Right → Down → Left**.

## Loot Crate

Next, we train on a more complex board configuration with crates but still without enemies. It learns how to successfully blow up crates. I also experimented with a 16-bit representation dropping the type of the object of interest to save 2 bits. However, the agent would get caught in infinite up-down loops or just wait in that reduced representation. My interpretation is that the agent benefits from being able to distinguish between coins and crates (even though I am not 100% sure why this should matter in tabular Q-learning). My takeaway is that the representation matters more that I thought at first.

<figure>
  <img src="{{ site.baseurl }}/assets/img/bomberle-python/5_crate_hero.gif"  width="100%" alt="">
  <figcaption>Figure 4: Tabular Q-agent successfully blowing up crates and collecting coins.</figcaption>
</figure>

## Free Game

In the final training phase, we simulate a full game with multiple agents:

* One **rule-based agent** (demonstrator)
* One **peaceful agent** that moves randomly and doesn't place bombs
* One **tabular Q-learning agent** in classic mode

<figure>
  <img src="{{ site.baseurl }}/assets/img/bomberle-python/6_allstar.gif"  width="100%" alt="">
  <figcaption>Figure 5: Tabular Q-agent (pink) playing against two rule-based and one peaceful agent.</figcaption>
</figure>
