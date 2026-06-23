# SPS Generative AI API - Assignment 2

This project extends the FastAPI and Docker application from Assignment 1 by adding a CIFAR10 image classifier implemented with PyTorch.

## What This Project Includes

- A spaCy word embedding endpoint from Assignment 1
- A CIFAR10 image classification endpoint using a CNN
- A PyTorch CNN model
- A training script for CIFAR10
- A trained model file
- Docker support
- Theory answers for CNN arithmetic questions

## CNN Architecture

The CNN follows the architecture required in Assignment 2:

1. Input RGB image resized to 64 x 64 x 3
2. Conv2D with 16 filters, kernel size 3 x 3, stride 1, padding 1
3. ReLU activation
4. MaxPooling2D with kernel size 2 x 2, stride 2
5. Conv2D with 32 filters, kernel size 3 x 3, stride 1, padding 1
6. ReLU activation
7. MaxPooling2D with kernel size 2 x 2, stride 2
8. Flatten
9. Fully connected layer with 100 units
10. ReLU activation
11. Fully connected layer with 10 output units

The CNN model is implemented in app/cifar10_model.py.

## Training

The training script is train_cifar10.py.

To train the CIFAR10 model, run:

uv run python train_cifar10.py

The script downloads CIFAR10, trains the CNN for one epoch, evaluates the model on the test set, and saves the trained model to models/cifar10_cnn.pth.

In my run, the model reached about 50% test accuracy after one epoch.

## Run the API Locally

Run:

uv run fastapi run main.py --host 127.0.0.1 --port 8000

Then open:

http://127.0.0.1:8000/docs

## API Endpoints

GET /

Returns basic API information.

GET /health

Returns the API status and whether the CIFAR10 model is available.

GET /embedding?word=apple

Returns the spaCy word embedding for the input word.

POST /classify-image

Accepts an uploaded image file and returns the predicted CIFAR10 class, predicted class index, confidence, and filename.

Example command:

curl.exe -X POST "http://127.0.0.1:8000/classify-image" -F "file=@sample_image.png"

Example response:

{"predicted_class":"cat","predicted_index":3,"confidence":0.5924044251441956,"filename":"sample_image.png"}

## Docker

Build the Docker image:

docker build -t sps-genai-hw2 .

Run the container:

docker run -p 8000:80 sps-genai-hw2

Then test the API at:

http://127.0.0.1:8000/docs

## Theory Questions

The answers to the CNN arithmetic and model.train() questions are in theory_answers.md.
