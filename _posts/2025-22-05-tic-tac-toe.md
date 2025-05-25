---
layout: post
title:  "Computers learning Tic-Tac-Toe - Pt. 1"
date:   2025-05-22
description: Learn how to make computers play Tic-Tac-Toe
---

<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

<p class="intro"><span class="dropcap">I</span>n today's post, we study ways to play Tic-Tac-Toe using the minimax and Q-learning algorithms. </p>

## Intro

This series of post introduces a number of (machine-learning) algorithms including the minimax algorithm, tabular Q-learning and deep Q-learning (using dual networks, duelling networks, convolutional networks and prioritised experience replay) to play the game Tic-Tac-Toe. It essentially reproduces <a href="https://github.com/fcarsten/tic-tac-toe/"> Carsten Friedrich's excellent notebook series</a>.  In today's posts we are looking at the minimax and Q-learning algorithms. I will gloss over some details, so please refer to Carsten's post for some more explanations. You may find the accompanying <a href="https://github.com/KunkelAlexander/tictactoe_rl"> Python code on GitHub</a>.

<img src="{{ site.baseurl }}/assets/img/tictactoe-python/1_minimax_vs_minimax_game_play.gif" alt="">


### Setup
I test the different algorithms using a $$3\times3$$-Tic-Tac-Toe board with two players. I implement every algorithm as a separate agent so that we can assess their performance by having them play against one another in different settings.


## Minimax algorithm
<img src="{{ site.baseurl }}/assets/img/tictactoe-python/2_minimax_vs_random_game_play.gif" alt="">

First, we turn towards a benchmarking algorithm: the <a href="https://en.wikipedia.org/wiki/Minimax">minimax algorithm</a>. A minimax agent assumes that its opponent will play optimally—making the best possible moves to win. Faced with such a formidable adversary, the minimax agent chooses moves that minimize its own loss — hence the name. If victory is out of reach, the agent aims to at least avoid defeat.

We will pitch the minimax agent against a random agent making random, legal moves to see how it performs.

<img src="{{ site.baseurl }}/assets/img/tictactoe-python/1_random_minimax_comparison.png" alt="">

The above chart shows the results of 10000 games between the different agents. When two random agents play against one another, it turns out that the first player making a move has a significant advantage. The minimax agent playing first against the random agent is therefore nearly unbeatable. Going second, it stumbles into draws 20% of the time, but will never lose. Finally, the minimax agent achieves 100% draws when playing against itself. You can marvel at its performance in the introductory animation.

But how does this black magic work? How does the minmax agent decide which move to make?

The minimax algorithm works by evaluating all possible future game states. In a game like Tic-Tac-Toe, this means exploring up to $$3^9=19683$$ board configurations—since each of the $$9$$ squares can be in one of three states: X, O, or empty. The algorithm assigns a score to each final state (win, loss, or draw), and then works backwards from these terminal states to the current position, assuming both players choose optimally at every turn. Ultimately, it selects the move that leads to the best achievable outcome.

The name "minimax" comes from this interplay: the agent tries to minimize the possible maximum loss, while the opponent tries to maximize its own gain. It’s a game of antagonistic optimization.

This sounds like a perfect strategy for all two-player, turn-based games, right? Well—yes, in theory. But the major drawback is computational cost. Even in a simple game like Tic-Tac-Toe, the number of potential board configurations (though reducible due to symmetries) is non-trivial. In complex games like chess, the explosion in possibilities is extreme. If we imagined each square having exactly $$6$$ possible states (empty, rook, bishop, knight, queen, pawn), we get $$6^{64}\propto 10^{49}$$ combinations.

To make minimax tractable in such cases, practical implementations use optimizations such as pruning (e.g., alpha-beta pruning) and heuristics to evaluate only a subset of promising positions.

And what can stop a minmax agent? Well, another minmax agent as we have seen above. But it turns out that we can also do better against a random agent. We will see how in the next section.


## Q-Learning

Now, that we know what performance an agent needs to achieve to be better than a random agent and to play optimally, we can turn towards <a href="https://en.wikipedia.org/wiki/Q-learning">Q-Learning</a>. While minimax works by exhaustively simulating all future outcomes, Q-learning takes a very different approach: instead of reasoning through every possible scenario, it learns from experience. More specifically, it learns which actions lead to better outcomes over time by interacting with the environment and updating a table of values—called a Q-table—that estimates the value of taking a given action in a given state.

In the context of Tic-Tac-Toe, a tabular Q-learning agent builds up a table where each entry maps:
- a board configuration (the state),
- and a move (the action),
- a numerical score: the Q-value.

These Q-values are updated as the agent plays games and receives feedback. The feedback is user-defined through rewards and can be adapted at different training stages. Over time, the agent learns to prefer actions that are more likely to lead to positive rewards, even without knowing all the possible outcomes of the game. This makes Q-learning especially powerful in larger environments where minimax is computationally infeasible.


We train a Q-learning agent by letting it play up to 10000 thousand games and observe how it gradually improves.

<img src="{{ site.baseurl }}/assets/img/tictactoe-python/2_random_vs_q.png" alt="">

Surprisingly, it achieves close to 90% victories playing second against the random agent whereas the minimax agent only achieved 80%. How is this possible?  Unlike minimax, which assumes the opponent plays perfectly, Q-learning can adapt to the behavior of different opponents—making it well-suited for environments where opponents are unpredictable or not fully rational. In this case,  it has observed that the random agent makes a lot of silly mistakes. And it takes full advantage of these by playing riskier than the minimax agent. That means it cannot avoid the occasional loss, but it performs better against the random agent. Interestingly, its risky strategies still seem to work against the minmax agent where it achieves 100% draws in my tests. Alternatively, we can also train the Q-agent directly with the minmax agent and it also learns how to play draws reliably.

So, what is going on under the hood? Let us take a look at the Q-table that we learned.

<iframe src="{{ site.baseurl }}/assets/img/tictactoe-python/3_q_table.html" width="100%" height="600" frameborder="0"></iframe>

There are many things to discover here. First of all, we observe that most of the table appears in orange. These are invalid states, e.g. two crosses and no naught, that the training algorithm did not visit. Therefore, the initial Q-values, which I set to $$0.6$$ were not updated. After roughly 15,000 invalid states, we encounter the first interesting state at position 15499: A position with exactly one cross in the bottom right.