---
layout: post
title:  "Computers learning PDEs Pt. 1: The linear Schrödinger equation"
date:   2026-09-27
description: Learn how to solve the Schrödinger-Poisson equation with physics-informed neural networks, starting with the linear Schrödinger equation!
---

<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

<p class="intro"><span class="dropcap">I</span>n today's post, we teach a neural network to solve the Schrödinger-Poisson equation using physics-informed neural networks (PINNs). As a first step, we look at the linear Schrödinger equation in one dimension.</p>


## Intro
The Schrödinger-Poisson equation describes a quantum wave function $\psi$ that moves in a potential $V$ which it creates itself through its own density $|\psi|^2$. It shows up in many places, for instance as a model for fuzzy dark matter in cosmology. Usually, one solves it with classical numerical methods such as finite differences or [Fourier methods][fd-post]. In this series, we try something different: We ask a neural network to learn the solution.

The idea of physics-informed neural networks was popularised by Raissi, Perdikaris and Karniadakis in their paper [Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations][pinn-paper]. The basic idea is simple. A neural network is a very flexible function. We can feed it a position $x$ and a time $t$ and ask it to output the value of the solution $\psi(x, t)$. At first, the network outputs nonsense. We then train it by penalising everything that is wrong with its output:
- The output does not satisfy the PDE.
- The output does not match the initial conditions.
- The output does not match the boundary conditions.

The trick that makes this work is automatic differentiation. Neural network libraries such as TensorFlow or Pytorch can compute exact derivatives of the network output with respect to its inputs. So we can plug the network directly into the PDE and measure how badly it violates it. Note that we do not need any solution data inside the domain. The network learns the solution from the physics alone, hence the name "physics-informed".

We will proceed in steps:
1. [The linear Schrödinger equation in space and time](#1-the-linear-schrödinger-equation): The network learns the full solution $\psi(x, t)$ on the whole space-time domain at once.
2. The linear Schrödinger equation with an implicit time discretisation: The network only learns the solution in space and we step forward in time.
3. The Schrödinger-Poisson equation

The implementation draws heavily on the [PINN code by Jan Blechschmidt][blechschmidt-code] (MIT license) accompanying the excellent review [Three ways to solve partial differential equations with neural networks][blechschmidt-paper].


## 1. The linear Schrödinger equation
Before we tackle the nonlinear problem, we start with the simplest possible case: The linear Schrödinger equation in one dimension. It reads

$$ i \hbar \partial_t \psi(x, t) = \left(-\frac{\hbar^2}{2m}\partial_x^2 + V(x, t)\right) \psi(x, t) $$

for $t \in [t_0, t_1]$ and $x \in [x_l, x_r]$. To fully specify the problem, we need an initial condition and boundary conditions on both sides of the domain:

$$ \psi(x, t_0) = \psi_0(x) \qquad \psi(x_l, t) = \psi_L(t) \qquad \psi(x_r, t) = \psi_R(t). $$

We work in dimensionless units with $\hbar = m = 1$ and set the potential $V = 0$. As a test problem, we choose a plane wave

$$ \psi(x, t) = e^{i(kx - \omega t)} \quad \text{with} \quad \omega = \frac{k^2}{2}, $$

with $k = 1$ on the domain $x \in [0, 1]$ and $t \in [0, \pi]$. The plane wave is an exact solution, so we can easily check how well the network does.

### Space and time in one go
Classical solvers usually march forward in time: They compute the solution at $t + \Delta t$ from the solution at $t$, step after step. In this first approach, we do something quite different. We treat time just like another spatial coordinate and solve the equation on the two-dimensional space-time domain $[x_l, x_r] \times [t_0, t_1]$ in one step. The network sees the whole history of the wave function at once. There are no time steps and no time discretisation at all. The initial condition is then simply a boundary condition on the $t = t_0$ edge of this 2D domain.

### Complex numbers
Neural networks usually work with real numbers, but the wave function is complex. The simplest fix is to let the network output two numbers: The real part and the imaginary part of $\psi$. So our network $\psi_\theta(x, t)$ takes two inputs $(x, t)$ and returns two outputs $(\mathrm{Re}\,\psi_\theta, \mathrm{Im}\,\psi_\theta)$. Here, $\theta$ stands for all the weights of the network.

### The residual
How do we tell the network that it should satisfy the Schrödinger equation? We move all terms to one side and define the residual

$$ r_{\theta}(x, t) = \left(i \partial_t + \frac{1}{2} \partial_x^2\right)\psi_{\theta}(x, t). $$

If $\psi_\theta$ were the exact solution, the residual would be zero everywhere. Splitting it into real and imaginary parts, we get two real equations. Up to an overall sign, these are

$$ \partial_t \mathrm{Re}\,\psi_\theta + \frac{1}{2}\partial_x^2 \mathrm{Im}\,\psi_\theta = 0 \qquad \partial_t \mathrm{Im}\,\psi_\theta - \frac{1}{2}\partial_x^2 \mathrm{Re}\,\psi_\theta = 0. $$

TensorFlow's `GradientTape` gives us the derivatives we need. For the second derivative in $x$, we simply nest two gradient computations:

{%- highlight python -%}
def compute_residual(model, xt):
    with tf.GradientTape(persistent=True) as tape:
        x, t = xt[:, 0:1], xt[:, 1:2]
        tape.watch(x)
        tape.watch(t)

        psi = model(tf.concat([x, t], axis=1))
        re, im = psi[:, 0], psi[:, 1]

        # First derivatives inside the tape so we can differentiate again
        re_x = tape.gradient(re, x)
        im_x = tape.gradient(im, x)

    re_t  = tape.gradient(re, t)
    im_t  = tape.gradient(im, t)
    re_xx = tape.gradient(re_x, x)
    im_xx = tape.gradient(im_x, x)
    del tape

    residual_re = re_t + 0.5 * im_xx
    residual_im = im_t - 0.5 * re_xx
    return tf.concat([residual_re, residual_im], axis=1)
{%- endhighlight -%}

### The loss function
We evaluate the residual at many randomly chosen points inside the domain, the so-called collocation points. The loss function then consists of three terms:

$$ \mathcal{L}(\theta) = \underbrace{\frac{1}{N_r}\sum_{j} |r_\theta(x_j, t_j)|^2}_{\text{PDE}} + \underbrace{\frac{1}{N_0}\sum_{j} |\psi_\theta(x_j, t_0) - \psi_0(x_j)|^2}_{\text{initial condition}} + \underbrace{\frac{1}{N_b}\sum_{j} |\psi_\theta(x_b, t_j) - \psi_b(t_j)|^2}_{\text{boundary conditions}} $$

In code, this is just three mean squared errors added together:

{%- highlight python -%}
def compute_loss(model, xt_collocation, xt_initial, psi_initial, xt_boundary, psi_boundary):
    loss  = tf.reduce_mean(tf.square(compute_residual(model, xt_collocation))) # PDE
    loss += tf.reduce_mean(tf.square(model(xt_initial)  - psi_initial))        # initial conditions
    loss += tf.reduce_mean(tf.square(model(xt_boundary) - psi_boundary))       # boundary conditions
    return loss
{%- endhighlight -%}

We use $50$ points for the initial condition, $50$ points on the boundaries and $10000$ collocation points, all drawn from uniform distributions. Note how little information the network actually gets about the solution: Only $100$ points on the edges of the space-time domain. Everything in between has to follow from the PDE.

<!-- TODO: figure of collocation points and initial/boundary data in the (t, x) plane -->

### The network
The network itself is a plain fully-connected feedforward network with $4$ hidden layers of $20$ neurons each and $\tanh$ activations. We choose $\tanh$ because it is smooth: We need to take second derivatives of the network, and activations like ReLU have vanishing second derivatives. The first layer rescales the inputs $(x, t)$ to $[-1, 1]$, which makes training easier.

{%- highlight python -%}
def init_model(num_hidden_layers=4, num_neurons_per_layer=20):
    model = tf.keras.Sequential()
    model.add(tf.keras.Input(2))
    # Normalise input to [-1, 1] in all dimensions
    model.add(tf.keras.layers.Lambda(
        lambda x: 2.0 * (x - lower_bounds) / (upper_bounds - lower_bounds) - 1.0
    ))
    for _ in range(num_hidden_layers):
        model.add(tf.keras.layers.Dense(num_neurons_per_layer, activation="tanh", kernel_initializer="glorot_normal"))
    model.add(tf.keras.layers.Dense(2)) # real and imaginary part
    return model
{%- endhighlight -%}

We train with the Adam optimiser for $5000$ steps and decrease the learning rate from $10^{-2}$ to $10^{-3}$ after $1000$ steps and to $5 \cdot 10^{-4}$ after $3000$ steps.

### Results
<!-- TODO: add figures (predicted vs. analytical initial/boundary data, real part in the domain, loss history) and quote the L1 errors on the collocation and test points -->

After training, the network reproduces the initial and boundary conditions. More interestingly, it also finds the correct plane wave inside the domain where it never saw any solution data. To make sure that the network did not simply memorise the collocation points, we also evaluate it on $10000$ fresh test points. The error there is just as low as on the collocation points.

This is encouraging, but the plane wave is also about the friendliest problem one can think of: It is smooth, has a single frequency and the domain is short. Solving for all times at once also has a downside: The network has to represent the entire space-time solution, which becomes hard for long times or rapidly oscillating waves. In the next part, we therefore go back to a more classical idea and combine the network with an implicit time discretisation. The network then only has to learn the solution in space, while we step forward in time.


[pinn-paper]: https://doi.org/10.1016/j.jcp.2018.10.045
[blechschmidt-code]: https://github.com/janblechschmidt/PDEsByNNs/
[blechschmidt-paper]: https://arxiv.org/abs/2102.11802
[fd-post]: https://kunkelalexander.github.io/blog/when-fourier-fails-finite-differences-post/
