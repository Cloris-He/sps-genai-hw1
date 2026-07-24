from fastapi import FastAPI, File, HTTPException, Query, UploadFile

from app.cifar10_model import CIFAR10Classifier
from app.diffusion_model import CIFAR10DiffusionGenerator
from app.embedding_model import EmbeddingModel
from app.energy_model import CIFAR10EnergyGenerator
from app.gan_model import MNISTGANGenerator


app = FastAPI(
    title="SPS Generative AI API",
    description=(
        "Generate word embeddings, classify CIFAR-10 images, "
        "and generate images with GAN, Energy-Based, and "
        "Diffusion models."
    ),
    version="4.0.0",
)


embedding_model = EmbeddingModel()
cifar10_classifier = CIFAR10Classifier()
mnist_gan_generator = MNISTGANGenerator()
cifar10_energy_generator = CIFAR10EnergyGenerator()
cifar10_diffusion_generator = CIFAR10DiffusionGenerator()


@app.get("/")
def read_root() -> dict:
    """Return basic information about the API."""
    return {
        "message": "Welcome to the SPS Generative AI API.",
        "docs": "/docs",
        "embedding_endpoint": "/embedding?word=apple",
        "image_classification_endpoint": "/classify-image",
        "mnist_gan_endpoint": "/generate-mnist?num_images=1",
        "energy_model_endpoint": (
            "/generate-energy?num_images=1&steps=100"
        ),
        "diffusion_model_endpoint": (
            "/generate-diffusion?"
            "num_images=1&diffusion_steps=20"
        ),
    }


@app.get("/health")
def health_check() -> dict:
    """Confirm that the API server and models are available."""
    return {
        "status": "healthy",
        "embedding_model": "en_core_web_lg",
        "cifar10_model_available":
            cifar10_classifier.model_available,
        "mnist_gan_model_available":
            mnist_gan_generator.model_available,
        "cifar10_energy_model_available":
            cifar10_energy_generator.model_available,
        "cifar10_diffusion_model_available":
            cifar10_diffusion_generator.model_available,
    }


@app.get("/embedding")
def get_embedding(
    word: str = Query(
        ...,
        description="A single word to embed",
        examples=["apple"],
    ),
) -> dict:
    """Return the embedding vector for one query word."""
    try:
        return embedding_model.get_embedding(word)
    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        ) from error


@app.post("/classify-image")
async def classify_image(
    file: UploadFile = File(...),
) -> dict:
    """Return the predicted CIFAR-10 class for an image."""
    try:
        image_bytes = await file.read()
        result = cifar10_classifier.predict(image_bytes)
        result["filename"] = file.filename
        return result
    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        ) from error


@app.get("/generate-mnist")
def generate_mnist(
    num_images: int = Query(
        1,
        ge=1,
        le=16,
        description=(
            "Number of MNIST-like digit images to generate."
        ),
    ),
) -> dict:
    """Generate MNIST-like images with the trained GAN."""
    try:
        return mnist_gan_generator.generate(
            num_images=num_images
        )
    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        ) from error


@app.get("/generate-energy")
def generate_energy_images(
    num_images: int = Query(
        1,
        ge=1,
        le=8,
        description=(
            "Number of CIFAR-10 images to generate."
        ),
    ),
    steps: int = Query(
        100,
        ge=1,
        le=256,
        description=(
            "Number of Langevin Dynamics sampling steps."
        ),
    ),
) -> dict:
    """Generate CIFAR-10 images with the Energy Model."""
    try:
        return cifar10_energy_generator.generate(
            num_images=num_images,
            steps=steps,
        )
    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        ) from error


@app.get("/generate-diffusion")
def generate_diffusion_images(
    num_images: int = Query(
        1,
        ge=1,
        le=8,
        description=(
            "Number of CIFAR-10 images to generate."
        ),
    ),
    diffusion_steps: int = Query(
        20,
        ge=1,
        le=256,
        description=(
            "Number of reverse diffusion steps."
        ),
    ),
) -> dict:
    """Generate CIFAR-10 images with the Diffusion Model."""
    try:
        return cifar10_diffusion_generator.generate(
            num_images=num_images,
            diffusion_steps=diffusion_steps,
        )
    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        ) from error
