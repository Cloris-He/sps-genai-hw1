import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.utils import save_image

from app.gan_model import Discriminator, Generator, MODEL_PATH, NOISE_DIM


def get_device():
    return (
        torch.device("mps")
        if torch.backends.mps.is_available()
        else torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    )


def train_gan(epochs: int, batch_size: int, learning_rate: float, max_batches: int | None):
    device = get_device()
    print(f"Using device: {device}")

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )

    dataset = datasets.MNIST(
        root="data",
        train=True,
        download=True,
        transform=transform,
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    generator = Generator().to(device)
    discriminator = Discriminator().to(device)

    criterion = nn.BCELoss()
    optimizer_generator = optim.Adam(
        generator.parameters(),
        lr=learning_rate,
        betas=(0.5, 0.999),
    )
    optimizer_discriminator = optim.Adam(
        discriminator.parameters(),
        lr=learning_rate,
        betas=(0.5, 0.999),
    )

    for epoch in range(epochs):
        for batch_number, (real_images, _) in enumerate(dataloader):
            if max_batches is not None and batch_number >= max_batches:
                break

            real_images = real_images.to(device)
            current_batch_size = real_images.size(0)

            real_labels = torch.ones(current_batch_size, 1).to(device)
            fake_labels = torch.zeros(current_batch_size, 1).to(device)

            noise = torch.randn(current_batch_size, NOISE_DIM).to(device)
            fake_images = generator(noise)

            discriminator_real_outputs = discriminator(real_images)
            discriminator_real_loss = criterion(discriminator_real_outputs, real_labels)

            discriminator_fake_outputs = discriminator(fake_images.detach())
            discriminator_fake_loss = criterion(discriminator_fake_outputs, fake_labels)

            discriminator_loss = (
                discriminator_real_loss + discriminator_fake_loss
            ) / 2

            discriminator.zero_grad()
            discriminator_loss.backward()
            optimizer_discriminator.step()

            noise = torch.randn(current_batch_size, NOISE_DIM).to(device)
            fake_images = generator(noise)
            generator_outputs = discriminator(fake_images)
            generator_loss = criterion(generator_outputs, real_labels)

            generator.zero_grad()
            generator_loss.backward()
            optimizer_generator.step()

            if batch_number % 100 == 0:
                print(
                    f"Epoch [{epoch + 1}/{epochs}] "
                    f"Batch [{batch_number}/{len(dataloader)}] "
                    f"D Loss: {discriminator_loss.item():.4f} "
                    f"G Loss: {generator_loss.item():.4f}"
                )

        with torch.no_grad():
            sample_noise = torch.randn(16, NOISE_DIM).to(device)
            sample_images = generator(sample_noise)
            sample_images = (sample_images + 1) / 2

        Path("models").mkdir(exist_ok=True)
        save_image(
            sample_images,
            "models/mnist_gan_samples.png",
            nrow=4,
        )

    Path("models").mkdir(exist_ok=True)
    torch.save(generator.state_dict(), MODEL_PATH)
    print(f"Saved generator model to {MODEL_PATH}")
    print("Saved sample images to models/mnist_gan_samples.png")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=0.0002)
    parser.add_argument("--max-batches", type=int, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_gan(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_batches=args.max_batches,
    )
