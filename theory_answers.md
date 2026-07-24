# Assignment 4 Theory Answers

## Question 1

Let the embedding dimension be `d`, let the maximum period be `P`, and let the dimension index be:

```math
i = 0, 1, \ldots, d-1
```

Using the sinusoidal convention from the course slide, the i-th dimension is:

```math
\text{Embedding}(t,i)
=
\begin{cases}
\sin\left(\dfrac{t}{P^{i/d}}\right),
& \text{if } i \text{ is even}
\\
\cos\left(\dfrac{t}{P^{(i-1)/d}}\right),
& \text{if } i \text{ is odd}
\end{cases}
```

Equivalently, for each pair of dimensions indexed by `k`:

```math
\text{Embedding}(t,2k)
=
\sin\left(\dfrac{t}{P^{2k/d}}\right)
```

```math
\text{Embedding}(t,2k+1)
=
\cos\left(\dfrac{t}{P^{2k/d}}\right)
```

The different dimensions use different frequencies, allowing the model to represent the timestep at multiple scales.

## Question 2

For:

```math
d = 8,\qquad t = 1,\qquad P = 10000
```

the four arguments shared by the sine/cosine pairs are:

```math
1,\qquad
\frac{1}{10000^{2/8}} = 0.1,\qquad
\frac{1}{10000^{4/8}} = 0.01,\qquad
\frac{1}{10000^{6/8}} = 0.001
```

Therefore, the sinusoidal embedding vector is:

```math
[
\sin(1),
\cos(1),
\sin(0.1),
\cos(0.1),
\sin(0.01),
\cos(0.01),
\sin(0.001),
\cos(0.001)
]
```

The approximate numerical values are:

```math
[
0.841471,
0.540302,
0.099833,
0.995004,
0.010000,
0.999950,
0.001000,
0.9999995
]
```

## Question 3

Positional encoding in Transformers and sinusoidal time embedding in diffusion models use the same general idea: they transform a scalar position or timestep into a fixed-dimensional vector containing sine and cosine values at several frequencies.

In a Transformer, positional encoding represents the location of each token in a sequence. It is usually added to the token embedding so the model can use token order while processing the tokens in parallel.

In a diffusion model, the sinusoidal embedding represents the current diffusion timestep or noise level. It is supplied to the denoising network so the network knows how much noise is present and which stage of denoising it should perform.

The key difference is that Transformer positional encoding represents token position, while diffusion time embedding represents the noise or denoising stage of an image.

## Question 4

Each stride-2 downsampling block divides both spatial dimensions by 2.

```math
64 \times 64
\rightarrow
32 \times 32
\rightarrow
16 \times 16
\rightarrow
8 \times 8
```

Therefore, the spatial resolution at the bottleneck is:

```math
8 \times 8
```

## Question 5

The UNet receives the noisy image $x_t$ and the timestep $t$. It outputs an estimate of the Gaussian noise that was added to the original image:

```math
\epsilon_\theta(x_t, t)
```

The output has the same shape as the input image.

During training, a noise tensor $\epsilon$ is sampled and used to create the noisy image $x_t$. The model prediction is then compared with the actual sampled noise.

The loss can be written as:

```math
L
=
\left\|
\epsilon -
\epsilon_\theta(x_t,t)
\right\|
```

The course implementation uses `L1Loss`, so it minimizes the mean absolute difference between the actual noise and the predicted noise. The loss is backpropagated to update the UNet parameters.

## Question 6: Basic Gradient Calculations

### a)

The function is:

```math
y = x^2 + 3x
```

Its derivative is:

```math
\frac{dy}{dx} = 2x + 3
```

At $x = 2$:

```math
\frac{dy}{dx} = 2(2) + 3 = 7
```

Therefore, the output is:

```text
x.grad = tensor([7.])
```

### b)

If `requires_grad=False`, PyTorch does not construct a computational graph for operations involving `x`.

Calling:

```python
y.backward()
```

raises an error because `y` does not require gradients and does not have a gradient function. Therefore, `x.grad` is not calculated.

### c)

No. When `torch.tensor()` is used without specifying `requires_grad`, the default value is `False`. Gradients are not tracked unless `requires_grad=True` is explicitly provided.

## Question 7: Introduce Weights

### a)

After adding:

```python
print("w.grad =", w.grad)
```

the result is:

```text
w.grad = None
```

This happens because `w` was created without `requires_grad=True`. PyTorch therefore does not track the operations with respect to `w`.

### b)

The code can be modified as follows:

```python
import torch

x = torch.tensor([2.0], requires_grad=True)
w = torch.tensor([1.0, 3.0], requires_grad=True)

y = w[0] * x**2 + w[1] * x

y.backward()

print("x.grad =", x.grad)
print("w.grad =", w.grad)
```

Since:

```math
y = w_0x^2 + w_1x
```

the gradients with respect to the weights are:

```math
\frac{\partial y}{\partial w_0} = x^2 = 4
```

```math
\frac{\partial y}{\partial w_1} = x = 2
```

The output is:

```text
x.grad = tensor([7.])
w.grad = tensor([4., 2.])
```

### c)

No. The default value of `requires_grad` for `torch.tensor()` is `False`, so gradients are not tracked unless it is explicitly set to `True`.

## Question 8: Breaking the Graph

The code fails because:

```python
z = y.detach()
```

removes `z` from the computational graph. The later value `w` therefore does not require gradients and has no path back to `x`. Calling `w.backward()` raises an error.

To keep using `z` while allowing gradients to flow back to `x`, use `clone()` without `detach()`:

```python
import torch

x = torch.tensor([1.0], requires_grad=True)

y = x * 3
z = y.clone()
w = z * 2

w.backward()

print("x.grad =", x.grad)
```

The derivative is:

```math
\frac{dw}{dx} = 3 \times 2 = 6
```

The output is:

```text
x.grad = tensor([6.])
```

Using `z = y` would also preserve the computational graph.

## Question 9: Gradient Accumulation

After the first backward call:

```text
After first backward: x.grad = tensor([2.])
```

After the second backward call:

```text
After second backward: x.grad = tensor([5.])
```

PyTorch accumulates gradients in `.grad` instead of replacing the existing value. The second gradient is `3`, so it is added to the first gradient of `2`:

```math
2 + 3 = 5
```

To avoid unwanted gradient accumulation, clear the gradient before the next backward call:

```python
x.grad.zero_()
```

For example:

```python
import torch

x = torch.tensor([1.0], requires_grad=True)

y1 = x * 2
y1.backward()
print("After first backward: x.grad =", x.grad)

x.grad.zero_()

y2 = x * 3
y2.backward()
print("After second backward: x.grad =", x.grad)
```

The second output is now:

```text
After second backward: x.grad = tensor([3.])
```

When training a neural network, the usual approach is to call:

```python
optimizer.zero_grad()
```

before each new backward pass.
