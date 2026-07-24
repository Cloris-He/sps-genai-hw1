import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from app.diffusion_model import (
    DiffusionModel,
    MODEL_PATH,
    UNet,
    offset_cosine_diffusion_schedule,
)


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def calculate_normalizer(train_dataset, batch_size, device):
    """Calculate per-channel mean and standard deviation."""
    stats_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    mean = torch.zeros(3)
    std = torch.zeros(3)
    total_samples = 0

    for images, _ in stats_loader:
        current_batch_size = images.size(0)
        images_flat = images.view(
            current_batch_size,
            3,
            -1,
        )

        batch_mean = images_flat.mean(
            dim=(0, 2)
        )
        batch_std = images_flat.std(
            dim=(0, 2)
        )

        mean += batch_mean * current_batch_size
        std += batch_std * current_batch_size
        total_samples += current_batch_size

    mean /= total_samples
    std /= total_samples

    mean = mean.reshape(
        1,
        3,
        1,
        1,
    ).to(device)

    std = std.reshape(
        1,
        3,
        1,
        1,
    ).to(device)

    return mean, std


def run_training_batch(
    model,
    images,
    optimizer,
    loss_function,
):
    images = (
        images - model.normalizer_mean
    ) / model.normalizer_std

    noises = torch.randn_like(images)

    diffusion_times = torch.rand(
        (
            images.size(0),
            1,
            1,
            1,
        ),
        device=images.device,
    )

    noise_rates, signal_rates = (
        offset_cosine_diffusion_schedule(
            diffusion_times
        )
    )

    noisy_images = (
        signal_rates * images
        + noise_rates * noises
    )

    predicted_noises, _ = model.denoise(
        noisy_images,
        noise_rates,
        signal_rates,
    )

    loss = loss_function(
        predicted_noises,
        noises,
    )

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()


def run_validation_batch(
    model,
    images,
    loss_function,
):
    images = (
        images - model.normalizer_mean
    ) / model.normalizer_std

    noises = torch.randn_like(images)

    diffusion_times = torch.rand(
        (
            images.size(0),
            1,
            1,
            1,
        ),
        device=images.device,
    )

    noise_rates, signal_rates = (
        offset_cosine_diffusion_schedule(
            diffusion_times
        )
    )

    noisy_images = (
        signal_rates * images
        + noise_rates * noises
    )

    with torch.no_grad():
        predicted_noises, _ = model.denoise(
            noisy_images,
            noise_rates,
            signal_rates,
        )

        loss = loss_function(
            predicted_noises,
            noises,
        )

    return loss.item()


def train_diffusion(
    epochs,
    batch_size,
    learning_rate,
    max_batches,
):
    torch.manual_seed(42)

    device = get_device()
    print(f"Using device: {device}")

    transform = transforms.ToTensor()

    train_dataset = datasets.CIFAR10(
        root="data",
        train=True,
        download=True,
        transform=transform,
    )

    test_dataset = datasets.CIFAR10(
        root="data",
        train=False,
        download=True,
        transform=transform,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    network = UNet()
    model = DiffusionModel(
        network
    ).to(device)

    mean, std = calculate_normalizer(
        train_dataset,
        batch_size,
        device,
    )

    model.set_normalizer(
        mean,
        std,
    )

    print(
        "Normalization mean:",
        mean.flatten().tolist(),
    )
    print(
        "Normalization std:",
        std.flatten().tolist(),
    )

    optimizer = torch.optim.AdamW(
        model.network.parameters(),
        lr=learning_rate,
        weight_decay=1e-4,
    )

    loss_function = nn.L1Loss()
    best_validation_loss = float("inf")

    MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    for epoch in range(epochs):
        model.train()
        training_losses = []

        for batch_number, (images, _) in enumerate(
            train_loader,
            start=1,
        ):
            images = images.to(device)

            loss = run_training_batch(
                model,
                images,
                optimizer,
                loss_function,
            )

            training_losses.append(loss)

            if (
                batch_number == 1
                or batch_number % 50 == 0
                or max_batches is not None
            ):
                print(
                    f"Epoch [{epoch + 1}/{epochs}] "
                    f"Train Batch "
                    f"[{batch_number}/{len(train_loader)}] "
                    f"Loss: {loss:.4f}"
                )

            if (
                max_batches is not None
                and batch_number >= max_batches
            ):
                break

        average_training_loss = (
            sum(training_losses)
            / len(training_losses)
        )

        model.eval()
        validation_losses = []

        for batch_number, (images, _) in enumerate(
            test_loader,
            start=1,
        ):
            images = images.to(device)

            loss = run_validation_batch(
                model,
                images,
                loss_function,
            )

            validation_losses.append(loss)

            if (
                max_batches is not None
                and batch_number >= max_batches
            ):
                break

        average_validation_loss = (
            sum(validation_losses)
            / len(validation_losses)
        )

        print(
            f"Epoch {epoch + 1} complete - "
            f"Train Loss: "
            f"{average_training_loss:.4f}, "
            f"Validation Loss: "
            f"{average_validation_loss:.4f}"
        )

        checkpoint = {
            "epoch": epoch + 1,
            "model_state_dict":
                model.network.state_dict(),
            "optimizer_state_dict":
                optimizer.state_dict(),
            "train_loss":
                average_training_loss,
            "validation_loss":
                average_validation_loss,
            "normalizer_mean":
                model.normalizer_mean.detach().cpu(),
            "normalizer_std":
                model.normalizer_std.detach().cpu(),
        }

        if (
            average_validation_loss
            < best_validation_loss
        ):
            best_validation_loss = (
                average_validation_loss
            )

            torch.save(
                checkpoint,
                MODEL_PATH,
            )

            print(
                f"Saved best model to {MODEL_PATH}"
            )


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Train the CIFAR-10 "
            "Diffusion Model."
        )
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.001,
    )

    parser.add_argument(
        "--max-batches",
        type=int,
        default=None,
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    train_diffusion(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_batches=args.max_batches,
    )
