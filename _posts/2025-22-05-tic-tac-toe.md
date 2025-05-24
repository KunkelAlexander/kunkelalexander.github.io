---
layout: post
title:  "Computers learning Tic-Tac-Toe"
date:   2025-05-22
description: Learn how to make computers play Tic-Tac-Toe
---

<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

<p class="intro"><span class="dropcap">I</span>n today's post, we study ways to play Tic-Tac-Toe using different (machine-learning) algorithms. </p>

## Intro

This post introduces a number of (machine-learning) algorithms including the minimax algorithm, tabular Q-learning and deep Q-learning (using dual networks, duelling networks, convolutional networks and prioritised experience replay) to play the game Tic-Tac-Toe. It essentially reproduces <a href="https://github.com/fcarsten/tic-tac-toe/"> Carsten Friedrich's excellent notebook series</a>.  I will gloss over some details, so please refer to Carsten's post for some more explanations.  To spice things up a bit, I also describe the profiling of the code using flame diagrams and hyperparameter optimisation using 'optuna'. You may find the accompanying <a href="https://github.com/KunkelAlexander/tictactoe_rl"> Python code on GitHub </a>.

<img src="{{ site.baseurl }}/assets/img/tictactoe-python/1_random_game_play.gif" alt="">


### Setup
I test the different algorithms using a $$3\times3$$-Tic-Tac-Toe board with two players. I implement every algorithm as a separate agent so that we can assess their performance by having them play against one another in different settings.

## Random play
<img src="{{ site.baseurl }}/assets/img/tictactoe-python/1_random_vs_random.png" alt="">

We first establish the baseline performance of an agent playing legal, random moves against itself. The above chart shows the results of $$3000$$ games for ten runs.  It turns out that the first player making a move has a significant advantage. This will be important to keep in mind for further benchmarking. The introductory animation shows what five typical games between two random agents might look like.


## Minimax algorithm
<img src="{{ site.baseurl }}/assets/img/tictactoe-python/2_minimax_vs_random_game_play.gif" alt="">

Now, we turn towards a different kind of benchmarking algorithm: the (minimax algorithm)[https://en.wikipedia.org/wiki/Minimax]. A minimax agent assumes that its opponent will play optimally—making the best possible moves to win. Faced with such a formidable adversary, the minimax agent chooses moves that minimize its own loss — hence the name. If victory is out of reach, the agent aims to at least avoid defeat.

But how does it decide which move to make?

The minimax algorithm works by evaluating all possible future game states. In a game like Tic-Tac-Toe, this means exploring up to $$3^9=19683$$ board configurations—since each of the $$9$$ squares can be in one of three states: X, O, or empty. The algorithm assigns a score to each final state (win, loss, or draw), and then works backwards from these terminal states to the current position, assuming both players choose optimally at every turn. Ultimately, it selects the move that leads to the best achievable outcome.

The name "minimax" comes from this interplay: the agent tries to minimize the possible maximum loss, while the opponent tries to maximize its own gain. It’s a game of antagonistic optimization.

This sounds like a perfect strategy for all two-player, turn-based games, right? Well—yes, in theory. But the major drawback is computational cost. Even in a simple game like Tic-Tac-Toe, the number of potential board configurations (though reducible due to symmetries) is non-trivial. In complex games like chess, the explosion in possibilities is extreme. If we imagined each square having exactly $$6$$ possible states (empty, rook, bishop, knight, queen, pawn), we get $$6^64\propto 10^{49}$$ combinations.

To make minimax tractable in such cases, practical implementations use optimizations such as pruning (e.g., alpha-beta pruning) and heuristics to evaluate only a subset of promising positions.

Returning to Tic-Tac-Toe: how well does a minimax agent perform against a random player?

Unsurprisingly, the minimax agent is nearly unbeatable. A random agent might occasionally stumble into a draw, but it has no chance of winning - a situation shown in the above animation. The minimax strategy guarantees either victory or a draw, depending on who moves first and how the game unfolds.

<img src="{{ site.baseurl }}/assets/img/tictactoe-python/2_minimax_vs_random.png" alt="">

Going first helps the random agent to achieve slightly more draws.

<img src="{{ site.baseurl }}/assets/img/tictactoe-python/3_random_vs_minimax.png" alt="">


But only another agent playing ideally manages to achieve the desired 100% draws.

<img src="{{ site.baseurl }}/assets/img/tictactoe-python/4_minimax_vs_minimax.png" alt="">

## Q-Learning

Now, that we know what performance an agent needs to achieve to be better than a random agent and to play optimally, we can turn towards (Q-learning)[https://en.wikipedia.org/wiki/Q-learning]. While Minimax works by exhaustively simulating all future outcomes, Q-learning takes a very different approach: instead of reasoning through every possible scenario, it learns from experience. More specifically, it learns which actions lead to better outcomes over time by interacting with the environment and updating a table of values—called a Q-table—that estimates the value of taking a given action in a given state.

In the context of Tic-Tac-Toe, a tabular Q-learning agent builds up a table where each entry maps:
- a board configuration (the state),
- and a move (the action),
- a numerical score: the Q-value.

These Q-values are updated as the agent plays games and receives feedback. The feedback is user-defined through rewards and can be adapted at different training stages. Over time, the agent learns to prefer actions that are more likely to lead to positive rewards, even without knowing all the possible outcomes of the game. This makes Q-learning especially powerful in larger environments where minimax is computationally infeasible.


In the next section, we’ll train a Q-learning agent by letting it play thousands of games and observe how it gradually improves.




Unlike minimax, which assumes the opponent plays perfectly, Q-learning can adapt to the behavior of different opponents—making it well-suited for environments where opponents are unpredictable or not fully rational.