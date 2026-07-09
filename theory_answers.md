# Assignment 3 Theory Answers

## Question 1

Given:

- Input size: I = 8
- Kernel size: K = 4
- Stride: S = 2
- Padding: P = 1
- Output padding: OP = 1

For a 2D transposed convolution, the output size is:

O = (I - 1)S - 2P + K + OP

Substitute the values:

O = (8 - 1)(2) - 2(1) + 4 + 1

O = 7(2) - 2 + 4 + 1

O = 14 - 2 + 4 + 1

O = 17

Therefore, the output feature map size is 17 x 17.

## Question 2

If the stride is increased from 2 to 3 while everything else stays fixed, the output size increases because the stride controls how far apart the transposed convolution places the expanded features.

Using the same values from Question 1, but with S = 3:

O = (I - 1)S - 2P + K + OP

O = (8 - 1)(3) - 2(1) + 4 + 1

O = 7(3) - 2 + 4 + 1

O = 21 - 2 + 4 + 1

O = 24

So the output size changes from 17 x 17 to 24 x 24. In general, increasing the stride increases the output size because the feature map is upsampled more aggressively.

## Question 3

The formula for the output size of a 2D transposed convolution is:

O = (I - 1)S - 2P + K + OP

where:

- I is the input size
- K is the kernel size
- S is the stride
- P is the padding
- OP is the output padding
- O is the output size

This formula is applied separately to height and width.

## Question 4

We want to upsample from 16 x 16 to 32 x 32, assuming no padding.

Using:

O = (I - 1)S - 2P + K + OP

Since there is no padding, P = 0. One possible configuration is:

- I = 16
- S = 2
- K = 2
- P = 0
- OP = 0

Substitute:

O = (16 - 1)(2) - 2(0) + 2 + 0

O = 15(2) + 2

O = 30 + 2

O = 32

Therefore, one possible configuration is kernel size 2 and stride 2 with no padding and no output padding.

## Question 5

Given the mini-batch:

[6, 8, 10, 6]

First compute the mean:

mean = (6 + 8 + 10 + 6) / 4

mean = 30 / 4

mean = 7.5

Next compute the variance using the batch variance:

variance = ((6 - 7.5)^2 + (8 - 7.5)^2 + (10 - 7.5)^2 + (6 - 7.5)^2) / 4

variance = ((-1.5)^2 + (0.5)^2 + (2.5)^2 + (-1.5)^2) / 4

variance = (2.25 + 0.25 + 6.25 + 2.25) / 4

variance = 11 / 4

variance = 2.75

The standard deviation is:

std = sqrt(2.75)

std ≈ 1.6583

Now normalize each value:

(6 - 7.5) / 1.6583 ≈ -0.9045

(8 - 7.5) / 1.6583 ≈ 0.3015

(10 - 7.5) / 1.6583 ≈ 1.5076

(6 - 7.5) / 1.6583 ≈ -0.9045

Therefore, the normalized output is approximately:

[-0.9045, 0.3015, 1.5076, -0.9045]

## Question 6

The key mathematical difference between ReLU and LeakyReLU is how they handle negative inputs.

For ReLU:

f(x) = max(0, x)

Equivalently:

f(x) = x, if x >= 0

f(x) = 0, if x < 0

For LeakyReLU:

f(x) = x, if x >= 0

f(x) = alpha x, if x < 0

where alpha is a small positive slope. In this assignment, the discriminator uses LeakyReLU(0.2), so alpha = 0.2.

ReLU sets all negative inputs to 0, while LeakyReLU keeps a small nonzero slope for negative inputs.

## Question 7

LeakyReLU may be preferred over ReLU in deep networks because it helps avoid the dying ReLU problem. With standard ReLU, if a neuron receives negative inputs for a long time, its output becomes 0 and its gradient can also become 0, so the neuron may stop learning.

LeakyReLU still allows a small gradient when the input is negative. This makes it easier for the network to keep updating weights during training. In GANs, this is especially useful for the discriminator because it helps maintain more stable learning when distinguishing real and fake samples.
