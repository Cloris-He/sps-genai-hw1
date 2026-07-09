from io import BytesIO
from pathlib import Path
import base64

from PIL import Image
import torch
import torch.nn as nn


NOISE_DIM = 100
MODEL_PATH = Path("models/mnist_gan_generator.pth")


class Generator(nn.Module):
    """Generator for MNIST images following the Assignment 3 architecture."""

    def __init__(self):
        super(Generator, self).__init__()
        self.fc = nn.Linear(NOISE_DIM, 7 * 7 * 128)
        self.deconv1 = nn.ConvTranspose2d(
            128, 64, kernel_size=4, stride=2, padding=1
        )
        self.batchnorm1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.deconv2 = nn.ConvTranspose2d(
            64, 1, kernel_size=4, stride=2, padding=1
        )
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.fc(x)
        x = x.view(x.size(0), 128, 7, 7)
        x = self.deconv1(x)
        x = self.batchnorm1(x)
        x = self.relu(x)
        x = self.deconv2(x)
        x = self.tanh(x)
        return x


class Discriminator(nn.Module):
    """Discriminator for MNIST images following the Assignment 3 architecture."""

    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(
            1, 64, kernel_size=4, stride=2, padding=1
        )
        self.leaky_relu1 = nn.LeakyReLU(0.2)
        self.conv2 = nn.Conv2d(
            64, 128, kernel_size=4, stride=2, padding=1
        )
        self.batchnorm2 = nn.BatchNorm2d(128)
        self.leaky_relu2 = nn.LeakyReLU(0.2)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(128 * 7 * 7, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.leaky_relu1(x)
        x = self.conv2(x)
        x = self.batchnorm2(x)
        x = self.leaky_relu2(x)
        x = self.flatten(x)
        x = self.fc(x)
        x = self.sigmoid(x)
        return x


class MNISTGANGenerator:
    """Load the trained generator and create MNIST-like digit images."""

    def __init__(self):
        self.device = (
            torch.device("mps")
            if torch.backends.mps.is_available()
            else torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )

        self.model = Generator().to(self.device)
        self.model_available = False

        if MODEL_PATH.exists():
            self.model.load_state_dict(torch.load(MODEL_PATH, map_location=self.device))
            self.model.eval()
            self.model_available = True

    def generate(self, num_images: int = 1) -> dict:
        if not self.model_available:
            raise ValueError("MNIST GAN generator file is missing. Please train the model first.")

        if num_images < 1 or num_images > 16:
            raise ValueError("num_images must be between 1 and 16.")

        noise = torch.randn(num_images, NOISE_DIM).to(self.device)

        self.model.eval()
        with torch.no_grad():
            generated_images = self.model(noise).cpu()

        generated_images = (generated_images + 1) / 2

        results = []
        for image_tensor in generated_images:
            image_array = (
                image_tensor.squeeze(0)
                .clamp(0, 1)
                .mul(255)
                .byte()
                .numpy()
            )
            image = Image.fromarray(image_array, mode="L")
            buffer = BytesIO()
            image.save(buffer, format="PNG")
            image_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
            results.append({"image_base64": image_base64})

        return {
            "num_images": num_images,
            "images": results,
        }
