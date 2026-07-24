from io import BytesIO
from pathlib import Path
import base64

from PIL import Image
import torch
import torch.nn as nn


IMAGE_SIZE = 32
NUM_CHANNELS = 3
MODEL_PATH = Path("models/cifar10_ebm.pth")


def swish(x):
    """Swish activation used in the course Energy Model example."""
    return x * torch.sigmoid(x)


class EnergyModel(nn.Module):
    """Convolutional Energy Model adapted from MNIST to CIFAR-10 images."""

    def __init__(self):
        super(EnergyModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2, padding=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 2 * 2, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = swish(self.conv1(x))
        x = swish(self.conv2(x))
        x = swish(self.conv3(x))
        x = swish(self.conv4(x))
        x = self.flatten(x)
        x = swish(self.fc1(x))
        return self.fc2(x)


def generate_samples(
    nn_energy_model,
    inp_imgs,
    steps=100,
    step_size=10.0,
    noise_std=0.01,
):
    """Use Langevin Dynamics to move input images toward low-energy states."""
    nn_energy_model.eval()

    for _ in range(steps):
        with torch.no_grad():
            noise = torch.randn_like(inp_imgs) * noise_std
            inp_imgs = (inp_imgs + noise).clamp(-1.0, 1.0)

        inp_imgs.requires_grad_(True)
        energy = nn_energy_model(inp_imgs)

        grads, = torch.autograd.grad(
            energy,
            inp_imgs,
            grad_outputs=torch.ones_like(energy),
        )

        with torch.no_grad():
            grads = grads.clamp(-0.03, 0.03)
            inp_imgs = (inp_imgs - step_size * grads).clamp(-1.0, 1.0)

    return inp_imgs.detach()


class CIFAR10EnergyGenerator:
    """Load the trained CIFAR-10 Energy Model and generate images."""

    def __init__(self):
        self.device = (
            torch.device("mps")
            if torch.backends.mps.is_available()
            else torch.device("cuda")
            if torch.cuda.is_available()
            else torch.device("cpu")
        )

        self.model = EnergyModel().to(self.device)
        self.model_available = False

        if MODEL_PATH.exists():
            self.model.load_state_dict(
                torch.load(MODEL_PATH, map_location=self.device)
            )
            self.model.eval()
            self.model_available = True

    def generate(self, num_images=1, steps=100):
        if not self.model_available:
            raise ValueError(
                "CIFAR-10 Energy Model file is missing. "
                "Please train the model first."
            )

        if num_images < 1 or num_images > 8:
            raise ValueError("num_images must be between 1 and 8.")

        if steps < 1 or steps > 256:
            raise ValueError("steps must be between 1 and 256.")

        initial_images = (
            torch.rand(
                (num_images, NUM_CHANNELS, IMAGE_SIZE, IMAGE_SIZE),
                device=self.device,
            )
            * 2
            - 1
        )

        with torch.enable_grad():
            generated_images = generate_samples(
                self.model,
                initial_images,
                steps=steps,
                step_size=10.0,
                noise_std=0.01,
            )

        generated_images = ((generated_images + 1) / 2).clamp(0, 1).cpu()

        results = []
        for image_tensor in generated_images:
            image_array = (
                image_tensor.permute(1, 2, 0)
                .mul(255)
                .byte()
                .numpy()
            )
            image = Image.fromarray(image_array)
            buffer = BytesIO()
            image.save(buffer, format="PNG")
            image_base64 = base64.b64encode(
                buffer.getvalue()
            ).decode("utf-8")
            results.append({"image_base64": image_base64})

        return {
            "model": "CIFAR-10 Energy Model",
            "num_images": num_images,
            "langevin_steps": steps,
            "images": results,
        }
