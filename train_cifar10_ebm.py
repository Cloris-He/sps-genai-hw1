import argparse
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from app.energy_model import EnergyModel, MODEL_PATH, generate_samples


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class Metric:
    """Track the average value of a training metric."""

    def __init__(self):
        self.reset()

    def update(self, value):
        self.total += value.item()
        self.count += 1

    def result(self):
        return self.total / self.count if self.count > 0 else 0.0

    def reset(self):
        self.total = 0.0
        self.count = 0


class Buffer:
    """Replay buffer adapted from the course EBM example to CIFAR-10."""

    def __init__(self, model, device, batch_size):
        self.model = model
        self.device = device
        self.batch_size = batch_size

        self.examples = [
            torch.rand((1, 3, 32, 32), device=device) * 2 - 1
            for _ in range(batch_size)
        ]

    def sample_new_examples(self, steps, step_size, noise):
        num_new = np.random.binomial(self.batch_size, 0.05)
        num_old = self.batch_size - num_new

        image_parts = []

        if num_new > 0:
            new_random_images = (
                torch.rand(
                    (num_new, 3, 32, 32),
                    device=self.device,
                )
                * 2
                - 1
            )
            image_parts.append(new_random_images)

        if num_old > 0:
            old_images = torch.cat(
                random.choices(self.examples, k=num_old),
                dim=0,
            )
            image_parts.append(old_images)

        input_images = torch.cat(image_parts, dim=0)

        new_images = generate_samples(
            self.model,
            input_images,
            steps=steps,
            step_size=step_size,
            noise_std=noise,
        )

        self.examples = (
            list(torch.split(new_images, 1, dim=0))
            + self.examples
        )
        self.examples = self.examples[:8192]

        return new_images


class EBM(nn.Module):
    """Energy-Based Model training wrapper from the course example."""

    def __init__(
        self,
        model,
        alpha,
        steps,
        step_size,
        noise,
        device,
        batch_size,
    ):
        super().__init__()

        self.model = model
        self.device = device
        self.alpha = alpha
        self.steps = steps
        self.step_size = step_size
        self.noise = noise

        self.buffer = Buffer(
            model=model,
            device=device,
            batch_size=batch_size,
        )

        self.loss_metric = Metric()
        self.reg_loss_metric = Metric()
        self.cdiv_loss_metric = Metric()
        self.real_out_metric = Metric()
        self.fake_out_metric = Metric()

    def metrics(self):
        return {
            "loss": self.loss_metric.result(),
            "reg": self.reg_loss_metric.result(),
            "cdiv": self.cdiv_loss_metric.result(),
            "real": self.real_out_metric.result(),
            "fake": self.fake_out_metric.result(),
        }

    def reset_metrics(self):
        for metric in [
            self.loss_metric,
            self.reg_loss_metric,
            self.cdiv_loss_metric,
            self.real_out_metric,
            self.fake_out_metric,
        ]:
            metric.reset()

    def train_step(self, real_images, optimizer):
        real_images = (
            real_images
            + torch.randn_like(real_images) * self.noise
        )
        real_images = torch.clamp(real_images, -1.0, 1.0)

        fake_images = self.buffer.sample_new_examples(
            steps=self.steps,
            step_size=self.step_size,
            noise=self.noise,
        )

        input_images = torch.cat(
            [real_images, fake_images],
            dim=0,
        )
        input_images = (
            input_images
            .clone()
            .detach()
            .to(self.device)
            .requires_grad_(False)
        )

        output_energies = self.model(input_images)

        real_energy, fake_energy = torch.split(
            output_energies,
            [real_images.size(0), fake_images.size(0)],
            dim=0,
        )

        contrastive_divergence_loss = (
            real_energy.mean() - fake_energy.mean()
        )

        regularization_loss = self.alpha * (
            real_energy.pow(2).mean()
            + fake_energy.pow(2).mean()
        )

        loss = (
            contrastive_divergence_loss
            + regularization_loss
        )

        optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            max_norm=0.1,
        )

        optimizer.step()

        self.loss_metric.update(loss)
        self.reg_loss_metric.update(regularization_loss)
        self.cdiv_loss_metric.update(
            contrastive_divergence_loss
        )
        self.real_out_metric.update(real_energy.mean())
        self.fake_out_metric.update(fake_energy.mean())

        return self.metrics()


def train_ebm(
    epochs,
    batch_size,
    learning_rate,
    sample_steps,
    max_batches,
):
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)

    device = get_device()
    print(f"Using device: {device}")

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                (0.5, 0.5, 0.5),
                (0.5, 0.5, 0.5),
            ),
        ]
    )

    train_dataset = datasets.CIFAR10(
        root="data",
        train=True,
        download=True,
        transform=transform,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,
    )

    energy_model = EnergyModel().to(device)

    ebm = EBM(
        model=energy_model,
        alpha=0.1,
        steps=sample_steps,
        step_size=10.0,
        noise=0.005,
        device=device,
        batch_size=batch_size,
    )

    optimizer = torch.optim.Adam(
        energy_model.parameters(),
        lr=learning_rate,
        betas=(0.0, 0.999),
    )

    for epoch in range(epochs):
        ebm.reset_metrics()

        for batch_number, (real_images, _) in enumerate(
            train_loader,
            start=1,
        ):
            real_images = real_images.to(device)

            metrics = ebm.train_step(
                real_images,
                optimizer,
            )

            if (
                batch_number == 1
                or batch_number % 50 == 0
                or max_batches is not None
            ):
                metric_text = ", ".join(
                    f"{name}: {value:.4f}"
                    for name, value in metrics.items()
                )

                print(
                    f"Epoch [{epoch + 1}/{epochs}] "
                    f"Batch [{batch_number}/{len(train_loader)}] "
                    f"{metric_text}"
                )

            if (
                max_batches is not None
                and batch_number >= max_batches
            ):
                break

        metric_text = ", ".join(
            f"{name}: {value:.4f}"
            for name, value in metrics.items()
        )
        print(
            f"Epoch {epoch + 1} complete - {metric_text}"
        )

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(energy_model.state_dict(), MODEL_PATH)
    print(f"Saved model to {MODEL_PATH}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train the CIFAR-10 Energy Model."
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.0001,
    )
    parser.add_argument(
        "--sample-steps",
        type=int,
        default=60,
    )
    parser.add_argument("--max-batches", type=int, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    train_ebm(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        sample_steps=args.sample_steps,
        max_batches=args.max_batches,
    )
