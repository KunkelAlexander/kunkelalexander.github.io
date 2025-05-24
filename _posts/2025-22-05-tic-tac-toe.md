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

<img src="{{ site.baseurl }}/assets/img/tictactoe-python/2_random_game_play.gif" alt="">


### Setup
I test the different algorithms using a $$3\times3$$-Tic-Tac-Toe board with two players. I implement every algorithm as a separate agent so that we can assess their performance by having them play against one another in different settings.

## Random play
<img src="{{ site.baseurl }}/assets/img/tictactoe-python/1_random.png" alt="">

We first establish the baseline performance of an agent playing legal, random moves against itself. The above chart shows the results of $$3000$$ games for ten runs.  It turns out that the first player making a move has a significant advantage. This will be important to keep in mind for further benchmarking. The introductory animation shows what five typical games between two random agents might look like.
