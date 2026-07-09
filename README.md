# SPS Generative AI API - Assignment 3

This project extends the FastAPI and Docker application from previous assignments by adding a Generative Adversarial Network (GAN) trained on the MNIST dataset.

The API supports word embeddings, CIFAR10 image classification, and MNIST-like handwritten digit generation.

## What This Project Includes

- A spaCy word embedding endpoint from Assignment 1
- A CIFAR10 image classification endpoint from Assignment 2
- A PyTorch GAN for generating MNIST-like handwritten digits
- A trained GAN generator model
- A FastAPI endpoint for generating MNIST images
- Docker support
- Theory answers for GAN building block questions

## GAN Architecture

The GAN follows the architecture required in Assignment 3.

### Generator

1. Input noise vector with shape (batch_size, 100)
2. Fully connected layer to 7 x 7 x 128
3. Reshape to (batch_size, 128, 7, 7)
4. ConvTranspose2D from 128 channels to 64 channels, kernel size 4, stride 2, padding 1
5. BatchNorm2D
6. ReLU activation
7. ConvTranspose2D from 64 channels to 1 channel, kernel size 4, stride 2, padding 1
8. Tanh activation
9. Output image with shape (batch_size, 1, 28, 28)

### Discriminator

1. Input image with shape (1, 28, 28)
2. Conv2D from 1 channel to 64 channels, kernel size 4, stride 2, padding 1
3. LeakyReLU(0.2)
4. Conv2D from 64 channels to 128 channels, kernel size 4, stride 2, padding 1
5. BatchNorm2D
6. LeakyReLU(0.2)
7. Flatten
8. Linear layer to a single output
9. Sigmoid activation for real/fake probability

The GAN model is implemented in app/gan_model.py.

## Training

The training script is train_mnist_gan.py.

To train the MNIST GAN, run:

uv run python train_mnist_gan.py --epochs 15 --batch-size 128

The script downloads MNIST, trains the generator and discriminator, saves the trained generator to models/mnist_gan_generator.pth, and saves generated sample images to models/mnist_gan_samples.png.

## Run the API Locally

Run:

uv run fastapi run main.py --host 127.0.0.1 --port 8000

Then open:

http://127.0.0.1:8000/docs

## API Endpoints

GET /
Returns basic API information.

GET /health
Returns the API status and whether the embedding model, CIFAR10 model, and MNIST GAN generator are available.

GET /embedding?word=apple
Returns the spaCy word embedding for the input word.

POST /classify-image
Accepts an uploaded image file and returns the predicted CIFAR10 class, predicted class index, confidence, and filename.

GET /generate-mnist?num_images=2
Generates MNIST-like handwritten digit images using the trained GAN generator. The response includes base64-encoded PNG images.

## Docker

Build the Docker image:

docker build -t sps-genai-hw3 .

Run the container:

docker run -p 8000:80 sps-genai-hw3

Then test the API at:

http://127.0.0.1:8000/docs

## Theory Questions

The answers to the GAN building block conceptual and calculation questions are in theory_answers.md.
