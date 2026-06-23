# Assignment 2 Theory Questions

## Question 1

Given:
- Input image size: 32 x 32 x 3
- Number of filters: 8
- Filter size: 5 x 5
- Stride: 1
- Padding: 0

The spatial output size is:

(32 - 5 + 2 * 0) / 1 + 1 = 28

Since there are 8 filters, the output depth is 8.

Answer: 28 x 28 x 8

## Question 2

If padding is changed to "same", the spatial size stays the same when stride is 1.

The input spatial size is 32 x 32, so the output spatial size is also 32 x 32.

Since there are 8 filters, the output depth is 8.

Answer: 32 x 32 x 8

## Question 3

Given:
- Input size: 64 x 64
- Filter size: 3 x 3
- Stride: 2
- Padding: 0

The spatial output size is:

floor((64 - 3 + 2 * 0) / 2) + 1
= floor(61 / 2) + 1
= 30 + 1
= 31

Answer: 31 x 31

## Question 4

Given:
- Input feature map size: 16 x 16
- Max-pooling size: 2 x 2
- Stride: 2

The spatial output size is:

(16 - 2) / 2 + 1 = 8

Answer: 8 x 8

## Question 5

The input image has shape 128 x 128.

Both convolutional layers use:
- Kernel size: 3 x 3
- Stride: 1
- Same padding

With same padding and stride 1, the spatial size stays the same after each convolutional layer.

After the first convolutional layer: 128 x 128

After the second convolutional layer: 128 x 128

Answer: 128 x 128

## Question 6

The command model.train() sets the model to training mode before the training loop.

If this line is removed, the effect depends on the model. For a simple CNN without dropout or batch normalization, there may be little or no visible difference because a new PyTorch model is usually already in training mode by default.

However, in general, model.train() is important because layers such as dropout and batch normalization behave differently during training and evaluation. Removing model.train() could cause the model to remain in evaluation mode if model.eval() was called earlier, which would make training behave incorrectly.

Answer: model.train() ensures the model is in training mode. Removing it may not affect this simple CNN much, but it can cause incorrect behavior for models with dropout, batch normalization, or models that were previously set to evaluation mode.
