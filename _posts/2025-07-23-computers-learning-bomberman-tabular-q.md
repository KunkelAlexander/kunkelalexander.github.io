---
layout: post
title:  "Computers learning Bomberman Pt. 1: Tabular Q-learning"
date:   2025-07-23
description: Learn how to make computers play Bomberman using the Q-learning algorithm
---

<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

<p class="intro"><span class="dropcap">I</span>n today's post, we use the tabular Q-learning algorithm to learn how to play a Bomberman clone Bomberle. Bomberle  https://hci.iwr.uni-heidelberg.de/vislearn/HTML/teaching/courses/FML/bomberman_rl/ https://github.com/ukoethe/bomberman_rl You may find the accompanying <a href="https://github.com/KunkelAlexander/bomberman_rl"> Python code on GitHub</a>.</p>

<img src="{{ site.baseurl }}/assets/img/bomberle-python/1_game_play.gif" alt="">




## Bomberle: Game Mechanics Explained

In the Bomberman clone we study, up to four players compete by collecting coins and placing bombs to destroy crates and eliminate enemies. The goal of the game is to collect as many coins and kill as many enemies as possible.

- Each player has **6 valid actions**: Move **up**, **right**, **down**, **left**, **Place a bomb**, **Wait**
- Walking into an obstacle results in **waiting** instead.
- To collect a coin, the player must **walk into it**.
- **Rocks**, **crates**, **bombs** (except immediately after placing your own) and **other players** are considered **obstacles** cannot be walked through
- Players can place **only one bomb at a time**. The timeline of a bomb is detailed below


| Frame | State             | Description                                      |
|-------|-------------------|--------------------------------------------------|
| 1     | Agent drops the bomb  | New bomb with time to explosion `t = BOMB_TIMER = 4`|
| 2     | Bomb ticking      | Bomb timer counts down to `t = 3`              |
| 3     | Bomb ticking      | Bomb timer counts down to `t = 2`              |
| 4     | Bomb ticking      | Bomb timer counts down to `t = 1`              |
| 5     | Bomb explodes     | Explosion at `t = 0`, affecting `BOMB_POWER = 3` tiles |
| 6     | Explosion remains | Explosion lingers for 1 frame (`EXPLOSION_TIMER = 2`) |
| 7+    | Agent can place a new bomb  | Harmless smoke remains for 2 frames |


<figure>

  <img src="{{ site.baseurl }}/assets/img/bomberle-python/2_explain_bombs.gif"  width="100%" alt="">
  <figcaption>Figure 1: Bomb game mechanics with default settings. The agent places a bomb, then moves away, waits for three frames and moves back into the smoke once the explosion has subsided.</figcaption>
</figure>


- **Coins**: +1 point per coin (set via `REWARD_COIN = 1` in `settings.py`)
- **Kills**: +5 points per enemy killed (set via `REWARD_KILL = 5`)
- If you commit suicide and kill another player simultaneously, you **still get 5 points**
- **Crate density**: Default is `0.75`, can be adjusted by scenario
- **Number of coins**: Default is `9`, can be scenario-specific
- **Board size**: Default is `13x13` (including walls), modified with `COLS` and `ROWS` parameters

All the settings can be found in the `settings.py`

## A good state representation for tabular Q-learning should be small

Could we pack the entire board state into a numeral representation? It turns out that this is unfeasible: assuming a small 9×9 board with 7×7−9 = 40 free fields that can either be free, occupied by a crate, a coin, an agent, or an explosion gives roughly $$4^{40} = 2^{80}\approx 10^{24}$$ states. And this does not even account for bomb availability, explosion timers, or bomb timers. A Q-table for six possible actions (`UP, RIGHT, DOWN, LEFT, BOMB, WAIT`), using 4-byte floating point numbers, would require around $$10^{10}$$ petabytes — clearly too much. Beyond memory requirements, trainability is another important consideration: a more granular state representation captures more nuances of the game, but becomes harder to train, since exotic edge states are visited infrequently and receive less training signal.

After some experimentation, I therefore settled for the following simplified representation of the game state using 22 bits:

| Bits      | Field                            | Description                                                                                  |
|-----------|----------------------------------|----------------------------------------------------------------------------------------------|
| **0–2**   | Neighbor (UP)                    | 3-bit code for object in UP direction: EMPTY, WALL, COIN, CRATE, ENEMY, BOMB, EXPLOSION     |
| **3–5**   | Neighbor (RIGHT)                 | 3-bit code for object in RIGHT direction                                                     |
| **6–8**   | Neighbor (DOWN)                  | 3-bit code for object in DOWN direction                                                      |
| **9–11**  | Neighbor (LEFT)                  | 3-bit code for object in LEFT direction                                                      |
| **12–13** | Direction to nearest object      | 2-bit code: `00` = UP, `01` = RIGHT, `10` = DOWN, `11` = LEFT                                |
| **14–15** | Type of nearest object           | 2-bit code: `00` = NONE, `01` = ENEMY, `10` = CRATE, `11` = COIN                             |
| **16**    | Action safe: UP                  | 1 = safe, 0 = unsafe                                                                         |
| **17**    | Action safe: RIGHT               | 1 = safe, 0 = unsafe                                                                         |
| **18**    | Action safe: DOWN                | 1 = safe, 0 = unsafe                                                                         |
| **19**    | Action safe: LEFT                | 1 = safe, 0 = unsafe                                                                         |
| **20**    | Action safe: BOMB                | 1 = safe, 0 = unsafe                                                                         |
| **21**    | Action safe: WAIT                | 1 = safe, 0 = unsafe                                                                         |


I use **3 bits per direction to encode the nearest neighbor** in each direction as one of the occupation states. This leaves one representation unused, which I consider acceptable. Unused states incur a memory penalty, but since they are never visited, they don’t waste CPU cycles. On the other hand, while introducing a new category — for instance, distinguishing between enemies with or without bombs, or bombs about to explode — might allow for a more accurate game-state model, it would slow down training. These 12 bits allow the agent to perceive its immediate surroundings. However, the agent still lacks any sense of the wider game board. To remedy this, I introduce **4 more bits that encode the direction and type of the nearest object of interest**. This encoding implies the agent can only pursue a single object of interest at a time. It will not be aware of, say, two coins at equal distance. If the agent is standing adjacent to the object of interest, the directional encoding will be inaccurate, since I chose not to include `WAIT` as a direction. But this is acceptable: the agent "sees" the object and is therefore in a clearly distinct state. This design choice should not negatively affect the performance of the tabular Q-agent.

Choosing the nearest object of interest is an art in itself, and highlights some of the challenges of tabular Q-learning. To ensure the agent receives useful information, we must compute features that even a rule-based agent would find useful — blurring the lines between pure learning and handcrafted logic. In practice, I’ve found that obtaining a good state representation always requires careful thought.

As for implementation, the code runs a breadth-first search (BFS) to locate the nearest coins, crates, and agents. If an agent is nearby (regardless of whether they’re behind crates or bombs), that agent is the object of interest. If no reachable agent exists, we guide the agent toward the nearest *walkable* agent. If none, then the nearest walkable coin. Failing that, the nearest walkable crate. If none of those are available, we fall back to the nearest agent, walkable or not.

Once the agent knows where to go, there remains a final question: what should it do? At this point, the agent has no sense of danger — and we need to fix that. So I add **6 more bits indicating whether each of the 6 actions is safe or unsafe**.

I define an action as "safe" if, taking into account all current bombs and explosions, executing that move still allows the agent to survive through the remaining blasts. That is, there exists at least one follow-up move or WAIT that avoids all danger until all bombs have exploded. This gives the agent a basic capacity to avoid walking into flames, trapping itself with its own bombs (a problem in previous designs), or failing to escape in time. It does *not* protect against being intentionally trapped by another agent or bombs being set off by other bombs, for instance. So this representation won’t lead to an ideal agent. But in practice, the trained agent should no longer kill itself, should reliably dodge explosions, break crates, collect coins, and occasionally kill another agent if it gets lucky. It will, however, still lose to a smart opponent that traps it in tunnels.

This representation consumes **22 bits in total**, translating to $$2^{22} = 4,194,304$$ possible states. A Q-table with 6 actions and 4-byte floats would then require about $$2^{22} * 6 * 4 \text{ bytes} = 100,663,296 \text{ bytes} \approx 100 \text{ megabytes}$$.

However, since many states are never visited during training, I store the Q-table as a Python dictionary. This trades a slight performance penalty on lookup for a dramatic reduction in memory use — down to just a few hundred kilobytes for a typical training run.

The animation below illustrates this encoding in practice. It shows the raw bit string, its division into the logical components described above, and the corresponding decoded categories. I found this reverse translation *extremely* helpful for debugging the encoding and would strongly recommend building a debug screen like this when designing state representations.


### A good reward system helps even when in-game rewards are scarce

Ideally, our reward system should mirror the games reward system yielding the same relative rewards for collecting coins and killing players. However, in practice rewards are sparse, especially when the player has to blow up crates to find coins and plays against strong opponents. Therefore, we introduce additional rewards that help the agent learn to explore the game.

I mirrow the rewards of the game with coin collected and killed opponent.
Additionally, the player is rewarded for surviving the round and destroying crates.
It gets punished for killing itself and getting killed and there is a slight penalty for inaction. The latter reflects that the enemy will have an easier time killing a waiting target even when the agent itself thinks that waiting is as good as  moving.

(Is there a case to be madew for the right normalisation of the rewards? What about the initialisation of the q table?
)
gent?


| Event              | Reward  | Description                              |
|--------------------|---------|------------------------------------------|
| COIN_COLLECTED     | +0.20   | Agent collected a coin                   |
| KILLED_OPPONENT    | +1.00   | Agent killed another agent               |
| CRATE_DESTROYED    | +0.10   | Agent destroyed a crate                  |
| KILLED_SELF        | −1.00   | Agent killed itself                      |
| SURVIVED_ROUND     | +1.00   | Agent survived until the end of the round|
| GOT_KILLED         | −0.10   | Agent got killed by opponent or bomb     |
| WAITED             | −0.02   | Agent performed a WAIT action            |

## Faster training through demonstration

We could just run games and update the Q-table simultaneously as we did with the Tic-Tac-Toe agent but I would like to speed up the training process to do some more thorough hyperparameter sweeps this time.
One thing we did not leverage for the Tic Tac Toe agent is that we had a demonstrator showing near-perfect play. The minimax agent. In the case of Bomberle, the developers kindly provide a rule-based agent that plays fairly well. We can leverage it to speed up the training process by recording a large number of transitions and updating the Q-table based on them before running the actual on-the-fly training with the agent. This allows the agent to explore states right from the start that it would otherwise only visit much later during training and hopefully speeds up the training.
What I do in pracrtice is record 50k transitions from the demonstrator and update the Q-table based on them before running the on-the-fly training and/or testing the agent's performance. For comparison: A regular training run with on-the-fly updates takes up to several hours (a number that could probably be cut down significantly with some optimisations that I might get back to later) whereas updating the q-table with the prerecorded 50k transitions only takes around 30 seconds, allowing for a much more thorough exploration of the hyperparameter space. The key speed gain comes from the fact that we can compute the rather expensive game state to integer state embedding only once and then reuse it with different hyperparameters.


## Coin heaven
We start the training in a simplified setting where the board is filled with gold coins
- Board size to 9x9
- no enemies
- no crates
- 100% coin density to begin with.

The figure below shows the exploration of different states during the training.

The below animation shows the agent's performance.

<figure>

  <img src="{{ site.baseurl }}/assets/img/bomberle-python/3_coin_grabber.gif"  width="100%" alt="">
  <figcaption>Figure 2: Bomb game mechanics with default settings. The agent places a bomb, then moves away, waits for three frames and moves back into the smoke once the explosion has subsided.</figcaption>
</figure>

The agent ends up collecting all coins, but its path is non-optimal: While it always collects one of the closest coins, it could save steps if it better planned its route. This is because the BFS algorithm only considers the next closest coin but not which series of moves might be optimal for more than one coin. Moreover, the agent's path highlight's a conscious choice that I made: The path-finding algorithm always prefers the closest object of interest that is in the Up, Left, Right, Down directions in exactly that order.

## Loot crate

We continue the trainign with with a board that contains crates filled with gold coins
- Board size to 9x9
- no enemies
- 75% crate density
- 100% coin density to begin with.


<figure>

  <img src="{{ site.baseurl }}/assets/img/bomberle-python/4_crate_hero.gif"  width="100%" alt="">
  <figcaption>Figure 3: Bomb game mechanics with default settings. The agent places a bomb, then moves away, waits for three frames and moves back into the smoke once the explosion has subsided.</figcaption>
</figure>


python3 main.py play --agents tql_demonstrator --train 1 --n-rounds 50000 --scenario coin-heaven --no-gui

cd agent_coude/tql_demonstrator

python3 build_training_episodes.py runs/coin_heaven_50k/transitions_all_games.pkl.gz /home/xerox/Documents/Programming/bomberman_rl/agent_code/tql_demonstrator/runs/coin_heaven_50k/transitions.pkl

python3 agent_code/tq_coingrabber/pretrain_agent.py --transitions-file agent_code/tql_demonstrator/runs/coin_heaven_50k/transitions.pkl agent_code/tq_coingrabber/q_table.npz


python3 main.py play --agents tql_demonstrator --train 1 --n-rounds 50000 --scenario loot-crate --no-gui

`python3 build_training_episodes.py runs/loot_crate_50k/transitions_all_games.pkl.gz  runs/loot_crate_50k/transitions.pkl`


python3 main.py play --agents tql_demonstrator peaceful_agent rule_based_agent rule_based_agent --train 1 --n-rounds 50000 --no-gui

python3 build_training_episodes.py runs/three_rule_based_peaceful_50k/transitions_all_games.pkl.gz  runs/three_rule_based_peaceful_50k/transitions.pkl



The above chart shows the results of 10000 games between the different agents. When two random agents play against one another, it turns out that the first player making a move has a significant advantage. The minimax agent playing first against the random agent is therefore nearly unbeatable. Going second, it stumbles into draws 20% of the time, but will never lose. Finally, the minimax agent achieves 100% draws when playing against itself. You can marvel at its performance in the introductory animation. But how does this black magic work?

### How does the minimax agent decide which move to make?

The minimax algorithm works by evaluating all possible future game states. In a game like Tic-Tac-Toe, this means exploring up to $$3^9=19683$$ board configurations—since each of the $$9$$ squares can be in one of three states: X, O, or empty. The algorithm assigns a score to each final state (win, loss, or draw), and then works backwards from these terminal states to the current position, assuming both players choose optimally at every turn. Ultimately, it selects the move that leads to the best achievable outcome.

The name `minimax` comes from this interplay: the agent tries to minimize the possible maximum loss, while the opponent tries to maximize its own gain. It’s a game of antagonistic optimization. In Tic-Tac-Toe, this equals to **assuming that the opponent will play perfectly and the best the agent can do is avoid defeat and aim for a draw**. This sounds like a perfect strategy for all two-player, turn-based games, right? Well—yes, in theory. But the major drawback is computational cost. Even in a simple game like Tic-Tac-Toe, the number of potential board configurations (though reducible due to symmetries) is non-trivial. In complex games like chess, the explosion in possibilities is extreme. If we imagined each square having exactly $$6$$ possible states (empty, rook, bishop, knight, queen, pawn), we get $$6^{64}\propto 10^{49}$$ combinations.

To make minimax tractable in such cases, practical implementations use optimizations such as pruning (e.g., alpha-beta pruning) and heuristics to evaluate only a subset of promising positions. And what can stop a minmax agent? Well, another minmax agent as we have seen above. But it turns out that we can also do better against a random agent. We will see how in the next section.
## Q-Learning

Now that we know what performance an agent needs to achieve to be better than a random agent and to play optimally, we can turn towards [Q-Learning](https://en.wikipedia.org/wiki/Q-learning). While minimax works by exhaustively simulating all future outcomes, Q-learning takes a very different approach: instead of reasoning through every possible scenario, it learns from experience. More specifically, it learns **how good a move is**, so it can choose the best one at each step. But how do we measure that?

We define the Q-value as the total expected reward an agent will get by taking an action in a given state and then always acting optimally after that. Let’s say you're in a certain board position  and you decide to place your mark in the center. You might not win immediately, but this move might lead to a win 3 turns later. The Q-value reflects that.

So, given a table where:
- **Columns** represent the different board configurations (the states),
- **Rows** represent the different moves (the actions),
- **Values** are the Q-values that tell you how good an action is in a given state

you just pick the action with the highest Q-value in a given state and you are guaranteed to maximise your expected reward.
Easy, right? Now, if you knew the Q-values from the start, things would really be easy. But unfortunately, you don't.

Rather, you start with a random table or just set all Q-values to the same value and update them as the agent plays games and receives feedback. Over time, the agent learns to prefer actions that are more likely to lead to positive rewards, even without knowing all possible outcomes of the game. This makes Q-learning especially powerful in larger environments where minimax is computationally infeasible.


### The Q-agent outperforms the minimax agent?

Before we dive into the details of Q-learning, let's train a Q-learning agent by letting it play up to 10,000 games against a random agent and observe how it gradually improves.

<figure>
  <img src="{{ site.baseurl }}/assets/img/tictactoe-python/2_random_vs_q.png" alt="Q-learning agent vs random agent performance">
  <figcaption>Figure 2: Performance of a Q-learning agent against a random opponent. After training, the Q-agent wins ~90% of games even when playing second.</figcaption>
</figure>


Surprisingly, it achieves close to **90% victories playing second against a random agent**, whereas the minimax agent only achieved 80%. How is this possible?

Unlike minimax, which assumes the opponent plays perfectly, Q-learning can adapt to the behavior of different opponents—making it well-suited for environments where opponents are unpredictable or not fully rational. In this case, it observed that the random agent makes many mistakes and it takes full advantage of these by playing riskier than the minimax agent. That means it cannot avoid the occasional loss, but it performs better overall.

Interestingly, its risky strategies still hold up against the minimax agent, achieving 100% draws. Alternatively, we can also train the Q-agent directly with the minimax agent, and it still learns to play draws reliably.

### How does the Q-agent learn about its environment?

The Q-value $$Q(s, a)$$, given a state $$s$$ where we take the action $$a$$, is updated using the **Bellman equation**:

$$
Q_{\text{new}}(s, a) = r + \gamma \cdot \max_{a'} Q(s', a'),
$$

where:
- $$s'$$: next state after taking the action,
- $$r$$: reward received after the transition,
- $$\gamma$$: discount factor $$ (0 \leq \gamma \leq 1) $$,
- $$\max_{a'} Q(s', a')$$: the estimated value of the best action in state $$s'$$.

This update nudges the old value of $$Q(s, a)$$ toward a **target value**:

$$
\text{Target} = r + \gamma \cdot \max_{a'} Q(s', a').
$$

Now, consider a case where the action $$a$$ taken in state $$s$$ leads directly to a win—meaning $$s'$$ is a terminal state and the game ends immediately. In this case, the agent receives a reward of $$r = +1$$. Since $$s'$$ is terminal, there are no valid next actions, so:

$$
\max_{a'} Q(s', a') = 0.
$$

Therefore, the Bellman update simplifies to:

$$
Q_{\text{new}}(s, a) = 1 + \gamma \cdot 0 = 1.
$$

In other words, the Q-value for a winning move is updated toward 1, reflecting the fact that this action directly results in the most favorable outcome. The agent will learn to strongly prefer this action in similar states in the future. This is a form of **bootstrapping**: the agent improves its estimates using its current estimates of future outcomes.

Can we give a theoretical justification for this update? Yes, we can.

We begin by assigning scalar rewards:
- **+1** for winning,
- **0** for drawing,
- **–1** for losing.

A rational player would try to win as often as possible and avoid losing, thereby maximizing the **cumulative reward** across a game.

A naive way to define the cumulative reward from time step $$t$$ is:

$$
R_t = \sum_{k=0}^{\infty} r_{t+k}.
$$

However, in more complex games (e.g. a shooter), there may be intermediate rewards—like damaging an opponent—which should not be valued equally with final outcomes. To account for this, Q-learning introduces **discounting** with a factor $$\gamma$$, defining the **discounted return** as:

$$
G_t = r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + \cdots = \sum_{k=0}^{\infty} \gamma^k r_{t+k}.
$$

The discount factor balances the trade-off between immediate and future rewards:
- $$\gamma = 0$$: only immediate rewards are considered.
- $$\gamma = 1$$: future rewards are valued equally with current ones.

In Tic-Tac-Toe, where rewards occur only at the end, setting $$\gamma = 0$$ means the agent receives no feedback for non-terminal moves. As a result, all intermediate actions seem equally ineffective. On the other hand, $$\gamma = 1$$ allows the agent to learn that certain early moves can lead to a win several steps later—making them highly valuable.

If we accept the cumulative return $$G_t$$ as a valid measure of long-term success, we can formally define the Q-value as:

$$
Q^\pi(s, a) = \mathbb{E}_\pi \left[ G_t \mid s_t = s, a_t = a \right],
$$

where $$\pi$$ is a **policy**—a rule for choosing actions (such as random play, greedy strategies, or minimax). The **optimal Q-value** is then:

$$
Q^*(s, a) = \max_\pi Q^\pi(s, a).
$$

Now, observe that the return $$G_t$$ is **recursive**:

$$
G_t = r_t + \gamma G_{t+1}.
$$

Substituting this into the definition of $$Q^\pi(s, a)$$ gives rise to the **Bellman equation**, which relates the value of a state-action pair to the expected reward plus the value of the next state. This recursive structure is what enables Q-learning to propagate final outcomes back to earlier decisions, allowing the agent to learn effective long-term strategies over time.


### What parameters do we need to worry about?

Tabular Q-learning introduces a number of hyperparameters:

- **Learning rate (α)**: Tabular Q-learning is not very sensistive to the learning rate and allows values closer to 1 (here: 0.01)
- **Training frequency**: I find that between 30,000 and 50,000 games with training every step work well.
- **Discount**: For Tic-Tac-Toe, there are only final and no intermediate rewards, so the discount is not independent from the rewards (here: 0.8).
- **Initial Q-values**: Higher initial Q-values foster more exploration, a value closer to zero leads to conservative and potentially slower training (here: 0.6)
- **Exploration**: During training, we do not always want the agent to strictly follow the Q-table. It can prove advantageous to let it randomly pick actions to further explore the state space (here: start with 100% exploration (random moves every move) that decreases by 1% every game and goes down to 0%-10% exploration)
### What do Q-tables look like?

We can gain deeper insights into the Q-agent's strategies by **visualising the Q-table as a heat map**.

<figure>
  <iframe src="{{ site.baseurl }}/assets/img/tictactoe-python/3_q_agent_q_table.html" width="100%" height="600" frameborder="0"></iframe>
  <figcaption>Figure 3: Q-table heatmap for an agent trained second against a random opponent. Brighter colors reflect higher Q-values. The agent favors center and corner plays unless blocked.</figcaption>
</figure>
What should you notice about the Q-agent's Q-table?
- **Invalid moves**: Invalid moves are indicated by white cells (occupied positions).
- **States with zero valid moves**: I excluded all states with zero moves in this visualisation.
    - Tic-Tac-Toe has $$3^9 = 19,683$$ states of which only $$5,478$$ are valid reachable states
    - The visualisation shows half of these reachable states since it only shows the Q-table for an agent playing second
- **Ordering**: States are ordered by number of valid moves left.
- **Initial Q-values**: Many entries are orange: this corresponds to the initial Q-value of $$0.6$$, chosen to balance exploration and conservatism.
- **Rarely visited states**: Early states with few or no visits such as the ones on the left retain their initial values.
- **Strong positions**: The agent strongly favors the center unless it's already taken—then it prefers corners.
- **Late game**: As more opponent moves are added, the agent learns to avoid traps and block wins.
- **Final game**: Final states may include Q-values of $$-1$$ (unavoidable loss), $$0$$ (draw), or $$1$$ (win).

We visualize the Q-values for a minimax agent for comparison. This can be done by visiting every state and asking the minimax algorithm what the ideal game outcome be for the player given its magnificent opponent.

<figure>
  <iframe src="{{ site.baseurl }}/assets/img/tictactoe-python/4_minimax_agent_q_table.html" width="100%" height="600" frameborder="0"></iframe>
  <figcaption>Figure 4: Q-values derived from a minimax agent playing second. The table reflects the agent’s pessimistic outlook, assuming the opponent plays perfectly. Many entries reflect inevitable draws or losses.</figcaption>
</figure>
How does the minimax agent's Q-table differ from the Q-agent's?
- **High pessimism**: The minimax agent assumes the opponent plays optimally. Accordingly, there are more zero or negative Q-values compared to the optimistic Q-agent trained on random play.
- **Late game**: Q-values reflect draws in most 8-move states.

Finally, training the Q-agent as second player against the minimax agent allows us to compare both tables more easily. For the sake of this experiment, we make the minimax agent **deterministic**: It always picks the same optimal action instead of randomly selecting one of the many optimal actions at its disposal in many states. This dramatically shrinks the number of states the Q-agent visits.

<figure>
  <iframe src="{{ site.baseurl }}/assets/img/tictactoe-python/5_det_minmax_vs_q_agent_q_table.html" width="100%" height="600" frameborder="0"></iframe>
  <figcaption>Figure 5: Q-table of an agent trained against a deterministic minimax opponent. The agent reliably learns to draw by mimicking optimal play under fixed conditions.</figcaption>
</figure>

The Q-table obtained this way is directly comparable to the minimax agent's Q-tables since the Q-agent was trained against an agent making optimal moves, i.e. it does not assume it can win games in case the opponent makes a mistake.

<figure>
  <iframe src="{{ site.baseurl }}/assets/img/tictactoe-python/6_det_minmax_vs_minmax_q_table.html" width="100%" height="600" frameborder="0"></iframe>
  <figcaption>Figure 6: Q-table for a minimax agent playing second against a deterministic version of itself. The Q-values confirm that every reachable state leads to a draw under perfect play.</figcaption>
</figure>

The minimax agent's and Q-agent's Q-tables match fairly well:
- **Shared pessimism**: Maximum Q-values are close to $$0$$ in both cases and match
- **Reliable draws**: Q-agent reliably has learned to reliably play draws against the deterministic minimax agent

## Conclusion

This post explored how the minimax and Q-learning algorithms can be used to teach computers how to play Tic-Tac-Toe. In the next post, we are going to look at Deep-Q-Learning approaches that replace the explicit Q-table by a regression algorithm using neural networks.

