from fastapi import FastAPI, File, HTTPException, Query, UploadFile

from app.cifar10_model import CIFAR10Classifier
from app.embedding_model import EmbeddingModel
from app.gan_model import MNISTGANGenerator

app = FastAPI(
    title="SPS Generative AI API",
    description="Generate word embeddings, classify CIFAR10 images, and generate MNIST digits with a GAN.",
    version="3.0.0",
)

embedding_model = EmbeddingModel()
cifar10_classifier = CIFAR10Classifier()
mnist_gan_generator = MNISTGANGenerator()


@app.get("/")
def read_root() -> dict:
    """Return basic information about the API."""
    return {
        "message": "Welcome to the SPS Generative AI API.",
        "docs": "/docs",
        "embedding_endpoint": "/embedding?word=apple",
        "image_classification_endpoint": "/classify-image",
        "mnist_gan_endpoint": "/generate-mnist?num_images=1",
    }


@app.get("/health")
def health_check() -> dict:
    """Confirm that the API server and models are available."""
    return {
        "status": "healthy",
        "embedding_model": "en_core_web_lg",
        "cifar10_model_available": cifar10_classifier.model_available,
        "mnist_gan_model_available": mnist_gan_generator.model_available,
    }


@app.get("/embedding")
def get_embedding(
    word: str = Query(..., description="A single word to embed", examples=["apple"]),
) -> dict:
    """Return the embedding vector for one query word."""
    try:
        return embedding_model.get_embedding(word)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@app.post("/classify-image")
async def classify_image(file: UploadFile = File(...)) -> dict:
    """Return the predicted CIFAR10 class for an uploaded image."""
    try:
        image_bytes = await file.read()
        result = cifar10_classifier.predict(image_bytes)
        result["filename"] = file.filename
        return result
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@app.get("/generate-mnist")
def generate_mnist(
    num_images: int = Query(
        1,
        ge=1,
        le=16,
        description="Number of MNIST-like digit images to generate.",
    ),
) -> dict:
    """Generate MNIST-like handwritten digit images using the trained GAN generator."""
    try:
        return mnist_gan_generator.generate(num_images=num_images)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
