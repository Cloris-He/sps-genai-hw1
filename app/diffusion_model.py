from io import BytesIO
from pathlib import Path
import base64
import math

from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F


IMAGE_SIZE = 32
NUM_CHANNELS = 3
MODEL_PATH = Path("models/cifar10_diffusion.pth")


def offset_cosine_diffusion_schedule(
    diffusion_times,
    min_signal_rate=0.02,
    max_signal_rate=0.95,
):
    """Return noise and signal rates for each diffusion time."""
    start_angle = torch.acos(
        torch.tensor(
            max_signal_rate,
            dtype=torch.float32,
            device=diffusion_times.device,
        )
    )
    end_angle = torch.acos(
        torch.tensor(
            min_signal_rate,
            dtype=torch.float32,
            device=diffusion_times.device,
        )
    )

    diffusion_angles = (
        start_angle
        + diffusion_times * (end_angle - start_angle)
    )

    signal_rates = torch.cos(diffusion_angles)
    noise_rates = torch.sin(diffusion_angles)

    return noise_rates, signal_rates


class SinusoidalEmbedding(nn.Module):
    """Sinusoidal embedding used in the course diffusion example."""

    def __init__(self, num_frequencies=16):
        super().__init__()
        self.num_frequencies = num_frequencies

        frequencies = torch.exp(
            torch.linspace(
                math.log(1.0),
                math.log(1000.0),
                num_frequencies,
            )
        )

        angular_speeds = (
            2.0
            * math.pi
            * frequencies.view(1, 1, 1, -1)
        )

        self.register_buffer(
            "angular_speeds",
            angular_speeds,
        )

    def forward(self, x):
        x = x.expand(
            -1,
            1,
            1,
            self.num_frequencies,
        )

        sin_part = torch.sin(
            self.angular_speeds * x
        )
        cos_part = torch.cos(
            self.angular_speeds * x
        )

        return torch.cat(
            [sin_part, cos_part],
            dim=-1,
        )


class ResidualBlock(nn.Module):
    """Residual convolution block from the course UNet."""

    def __init__(self, in_channels, out_channels):
        super().__init__()

        if in_channels != out_channels:
            self.projection = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
            )
        else:
            self.projection = nn.Identity()

        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
        )
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
        )

    def forward(self, x):
        residual = self.projection(x)

        x = self.conv1(x)
        x = x * torch.sigmoid(x)
        x = self.conv2(x)

        return x + residual


class DownBlock(nn.Module):
    """Residual blocks followed by average pooling."""

    def __init__(
        self,
        width,
        block_depth,
        in_channels,
    ):
        super().__init__()

        self.blocks = nn.ModuleList()

        for _ in range(block_depth):
            self.blocks.append(
                ResidualBlock(
                    in_channels,
                    width,
                )
            )
            in_channels = width

        self.pool = nn.AvgPool2d(
            kernel_size=2
        )

    def forward(self, x, skips):
        for block in self.blocks:
            x = block(x)
            skips.append(x)

        return self.pool(x)


class UpBlock(nn.Module):
    """Upsampling blocks with UNet skip connections."""

    def __init__(
        self,
        width,
        block_depth,
        in_channels,
    ):
        super().__init__()

        self.blocks = nn.ModuleList()

        for _ in range(block_depth):
            self.blocks.append(
                ResidualBlock(
                    in_channels + width,
                    width,
                )
            )
            in_channels = width

    def forward(self, x, skips):
        x = F.interpolate(
            x,
            scale_factor=2,
            mode="bilinear",
            align_corners=False,
        )

        for block in self.blocks:
            skip = skips.pop()
            x = torch.cat(
                [x, skip],
                dim=1,
            )
            x = block(x)

        return x


class UNet(nn.Module):
    """UNet adapted from 64x64 images to CIFAR-10 32x32 images."""

    def __init__(
        self,
        image_size=IMAGE_SIZE,
        num_channels=NUM_CHANNELS,
    ):
        super().__init__()

        self.image_size = image_size
        self.num_channels = num_channels

        self.initial = nn.Conv2d(
            num_channels,
            32,
            kernel_size=1,
        )

        self.embedding = SinusoidalEmbedding(
            num_frequencies=16
        )

        self.down1 = DownBlock(
            width=32,
            block_depth=2,
            in_channels=64,
        )
        self.down2 = DownBlock(
            width=64,
            block_depth=2,
            in_channels=32,
        )
        self.down3 = DownBlock(
            width=96,
            block_depth=2,
            in_channels=64,
        )

        self.mid1 = ResidualBlock(
            in_channels=96,
            out_channels=128,
        )
        self.mid2 = ResidualBlock(
            in_channels=128,
            out_channels=128,
        )

        self.up1 = UpBlock(
            width=96,
            block_depth=2,
            in_channels=128,
        )
        self.up2 = UpBlock(
            width=64,
            block_depth=2,
            in_channels=96,
        )
        self.up3 = UpBlock(
            width=32,
            block_depth=2,
            in_channels=64,
        )

        self.final = nn.Conv2d(
            32,
            num_channels,
            kernel_size=1,
        )

        nn.init.zeros_(self.final.weight)

    def forward(
        self,
        noisy_images,
        noise_variances,
    ):
        skips = []

        x = self.initial(noisy_images)

        noise_embedding = self.embedding(
            noise_variances
        )

        noise_embedding = noise_embedding.permute(
            0,
            3,
            1,
            2,
        )

        noise_embedding = F.interpolate(
            noise_embedding,
            size=(
                self.image_size,
                self.image_size,
            ),
            mode="nearest",
        )

        x = torch.cat(
            [x, noise_embedding],
            dim=1,
        )

        x = self.down1(x, skips)
        x = self.down2(x, skips)
        x = self.down3(x, skips)

        x = self.mid1(x)
        x = self.mid2(x)

        x = self.up1(x, skips)
        x = self.up2(x, skips)
        x = self.up3(x, skips)

        return self.final(x)


class DiffusionModel(nn.Module):
    """Diffusion process and reverse denoising process."""

    def __init__(self, network):
        super().__init__()

        self.network = network

        self.register_buffer(
            "normalizer_mean",
            torch.zeros(
                1,
                NUM_CHANNELS,
                1,
                1,
            ),
        )
        self.register_buffer(
            "normalizer_std",
            torch.ones(
                1,
                NUM_CHANNELS,
                1,
                1,
            ),
        )

    def set_normalizer(self, mean, std):
        self.normalizer_mean.copy_(mean)
        self.normalizer_std.copy_(std)

    def denormalize(self, images):
        images = (
            images * self.normalizer_std
            + self.normalizer_mean
        )

        return torch.clamp(
            images,
            0.0,
            1.0,
        )

    def denoise(
        self,
        noisy_images,
        noise_rates,
        signal_rates,
    ):
        predicted_noises = self.network(
            noisy_images,
            noise_rates ** 2,
        )

        predicted_images = (
            noisy_images
            - noise_rates * predicted_noises
        ) / signal_rates

        return predicted_noises, predicted_images

    def reverse_diffusion(
        self,
        initial_noise,
        diffusion_steps,
    ):
        step_size = 1.0 / diffusion_steps
        current_images = initial_noise

        for step in range(diffusion_steps):
            diffusion_times = torch.ones(
                (
                    initial_noise.size(0),
                    1,
                    1,
                    1,
                ),
                device=initial_noise.device,
            )

            diffusion_times = diffusion_times * (
                1.0 - step * step_size
            )

            noise_rates, signal_rates = (
                offset_cosine_diffusion_schedule(
                    diffusion_times
                )
            )

            (
                predicted_noises,
                predicted_images,
            ) = self.denoise(
                current_images,
                noise_rates,
                signal_rates,
            )

            next_diffusion_times = (
                diffusion_times - step_size
            )

            (
                next_noise_rates,
                next_signal_rates,
            ) = offset_cosine_diffusion_schedule(
                next_diffusion_times
            )

            current_images = (
                next_signal_rates
                * predicted_images
                + next_noise_rates
                * predicted_noises
            )

        return predicted_images

    def generate(
        self,
        num_images,
        diffusion_steps,
    ):
        device = next(
            self.parameters()
        ).device

        initial_noise = torch.randn(
            (
                num_images,
                NUM_CHANNELS,
                IMAGE_SIZE,
                IMAGE_SIZE,
            ),
            device=device,
        )

        self.eval()

        with torch.no_grad():
            generated_images = (
                self.reverse_diffusion(
                    initial_noise,
                    diffusion_steps,
                )
            )

        return self.denormalize(
            generated_images
        )


class CIFAR10DiffusionGenerator:
    """Load the trained diffusion model and generate CIFAR-10 images."""

    def __init__(self):
        self.device = (
            torch.device("mps")
            if torch.backends.mps.is_available()
            else torch.device("cuda")
            if torch.cuda.is_available()
            else torch.device("cpu")
        )

        network = UNet()
        self.model = DiffusionModel(
            network
        ).to(self.device)

        self.model_available = False

        if MODEL_PATH.exists():
            checkpoint = torch.load(
                MODEL_PATH,
                map_location=self.device,
            )

            if (
                isinstance(checkpoint, dict)
                and "model_state_dict" in checkpoint
            ):
                self.model.network.load_state_dict(
                    checkpoint[
                        "model_state_dict"
                    ]
                )

                self.model.set_normalizer(
                    checkpoint[
                        "normalizer_mean"
                    ].to(self.device),
                    checkpoint[
                        "normalizer_std"
                    ].to(self.device),
                )
            else:
                self.model.network.load_state_dict(
                    checkpoint
                )

            self.model.eval()
            self.model_available = True

    def generate(
        self,
        num_images=1,
        diffusion_steps=20,
    ):
        if not self.model_available:
            raise ValueError(
                "CIFAR-10 Diffusion Model file is missing. "
                "Please train the model first."
            )

        if num_images < 1 or num_images > 8:
            raise ValueError(
                "num_images must be between 1 and 8."
            )

        if (
            diffusion_steps < 1
            or diffusion_steps > 256
        ):
            raise ValueError(
                "diffusion_steps must be between 1 and 256."
            )

        generated_images = self.model.generate(
            num_images=num_images,
            diffusion_steps=diffusion_steps,
        ).cpu()

        results = []

        for image_tensor in generated_images:
            image_array = (
                image_tensor
                .permute(1, 2, 0)
                .mul(255)
                .byte()
                .numpy()
            )

            image = Image.fromarray(
                image_array
            )

            buffer = BytesIO()
            image.save(
                buffer,
                format="PNG",
            )

            image_base64 = base64.b64encode(
                buffer.getvalue()
            ).decode("utf-8")

            results.append(
                {
                    "image_base64": image_base64
                }
            )

        return {
            "model": "CIFAR-10 Diffusion Model",
            "num_images": num_images,
            "diffusion_steps": diffusion_steps,
            "images": results,
        }
