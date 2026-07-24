# SPS Generative AI API - Assignment 4

This project extends the FastAPI and Docker application from previous assignments by adding an Energy-Based Model and a Diffusion Model trained on the CIFAR-10 dataset.

The API supports word embeddings, CIFAR-10 image classification, MNIST-like handwritten digit generation, and CIFAR-10 image generation with Energy-Based and Diffusion models.

## What This Project Includes

- A spaCy word embedding endpoint from Assignment 1
- A CIFAR-10 image classification endpoint from Assignment 2
- A PyTorch GAN for generating MNIST-like handwritten digits from Assignment 3
- A PyTorch Energy-Based Model trained on CIFAR-10
- Langevin Dynamics for sampling low-energy CIFAR-10 images
- A PyTorch Diffusion Model trained on CIFAR-10
- A UNet with sinusoidal noise embeddings for predicting added noise
- FastAPI endpoints for running both new image generation models
- Trained model checkpoint files
- Docker support
- Theory answers for diffusion models, energy models, and PyTorch gradients

## Energy-Based Model

The Energy-Based Model is implemented in `app/energy_model.py`.

The model:

1. Accepts CIFAR-10 images with shape `(batch_size, 3, 32, 32)`
2. Applies four convolutional layers with Swish activations
3. Flattens the convolutional output
4. Uses fully connected layers to produce one energy value for each image

Low-energy images are generated with Langevin Dynamics. The input images are initialized with random noise and repeatedly updated using gradients of the energy with respect to the input pixels.

The trained model is stored in:

`models/cifar10_ebm.pth`

## Diffusion Model

The Diffusion Model is implemented in `app/diffusion_model.py`.

The model includes:

1. A sinusoidal embedding for the diffusion noise level
2. Residual convolution blocks
3. Three downsampling blocks using average pooling
4. Bottleneck residual blocks
5. Three upsampling blocks with UNet skip connections
6. A final convolutional layer that predicts the Gaussian noise added to the image

During training, CIFAR-10 images are corrupted with Gaussian noise at randomly selected diffusion times. The UNet predicts the added noise using an L1 loss.

During image generation, the model begins with Gaussian noise and iteratively applies the reverse diffusion process.

The trained model is stored in:

`models/cifar10_diffusion.pth`

## Training

The trained model checkpoints are included, so retraining is not required to run the API.

### Train the Energy-Based Model

The training script is:

`train_cifar10_ebm.py`

Run:

`uv run python train_cifar10_ebm.py --epochs 10 --batch-size 128 --sample-steps 60`

The script downloads CIFAR-10, trains the Energy-Based Model, and saves the checkpoint to `models/cifar10_ebm.pth`.

### Train the Diffusion Model

The training script is:

`train_cifar10_diffusion.py`

Run:

`uv run python train_cifar10_diffusion.py --epochs 10 --batch-size 64 --learning-rate 0.001`

The script downloads CIFAR-10, trains the Diffusion Model, evaluates validation loss, and saves the best checkpoint to `models/cifar10_diffusion.pth`.

The training scripts automatically use MPS, CUDA, or CPU depending on what is available.

## Run the API Locally

Run:

`uv run fastapi run main.py --host 127.0.0.1 --port 8000`

Then open:

`http://127.0.0.1:8000/docs`

## API Endpoints

### `GET /`

Returns basic API information and example endpoint URLs.

### `GET /health`

Returns the API status and whether all trained models are available.

### `GET /embedding?word=apple`

Returns the spaCy word embedding for the input word.

### `POST /classify-image`

Accepts an uploaded image and returns the predicted CIFAR-10 class, predicted class index, confidence, and filename.

### `GET /generate-mnist?num_images=2`

Generates MNIST-like handwritten digit images using the trained GAN generator.

### `GET /generate-energy?num_images=1&steps=100`

Generates CIFAR-10 images using the trained Energy-Based Model and Langevin Dynamics.

Parameters:

- `num_images`: number of images to generate, from 1 to 8
- `steps`: number of Langevin Dynamics sampling steps, from 1 to 256

### `GET /generate-diffusion?num_images=1&diffusion_steps=20`

Generates CIFAR-10 images using the trained Diffusion Model.

Parameters:

- `num_images`: number of images to generate, from 1 to 8
- `diffusion_steps`: number of reverse diffusion steps, from 1 to 256

The image generation endpoints return base64-encoded PNG images.

## Docker

Build the Docker image:

`docker build -t sps-genai-hw4 .`

Run the container:

`docker run --rm -p 8000:80 sps-genai-hw4`

Then open:

`http://127.0.0.1:8000/docs`

The Docker application runs on CPU when CUDA or MPS is unavailable.

## Project Files

- `main.py`: FastAPI application and endpoints
- `app/embedding_model.py`: spaCy word embeddings
- `app/cifar10_model.py`: CIFAR-10 image classifier
- `app/gan_model.py`: MNIST GAN generator
- `app/energy_model.py`: CIFAR-10 Energy-Based Model and Langevin sampling
- `app/diffusion_model.py`: CIFAR-10 UNet and diffusion process
- `train_cifar10.py`: CIFAR-10 classifier training
- `train_mnist_gan.py`: GAN training
- `train_cifar10_ebm.py`: Energy-Based Model training
- `train_cifar10_diffusion.py`: Diffusion Model training
- `models/`: trained model checkpoints
- `theory_answers.md`: answers to the Assignment 4 theory questions

## Theory Questions

The answers to the diffusion model, energy model, and PyTorch gradient questions are in `theory_answers.md`.
