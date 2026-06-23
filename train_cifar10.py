import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from app.cifar10_model import CIFAR10CNN, MODEL_PATH


torch.manual_seed(42)
np.random.seed(42)

BATCH_SIZE = 32
EPOCHS = 1


device = (
    torch.device("mps")
    if torch.backends.mps.is_available()
    else torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
)

print(f"Using device: {device}")

transform = transforms.Compose(
    [
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ]
)

train_dataset = torchvision.datasets.CIFAR10(
    root="./data", train=True, download=True, transform=transform
)

test_dataset = torchvision.datasets.CIFAR10(
    root="./data", train=False, download=True, transform=transform
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

model = CIFAR10CNN().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0005)

for epoch in range(EPOCHS):
    running_loss = 0.0
    running_correct = 0
    running_total = 0

    model.train()

    for batch_number, (inputs, labels) in enumerate(train_loader):
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_correct += (predicted == labels).sum().item()
        running_total += labels.size(0)
        running_loss += loss.item()

        if batch_number % 100 == 99:
            print(
                f"Epoch {epoch + 1}, Batch {batch_number + 1}, "
                f"Avg Loss: {running_loss / (batch_number + 1):.4f}, "
                f"Avg Accuracy: {running_correct / running_total:.3f}"
            )

test_correct = 0
test_total = 0

model.eval()
with torch.no_grad():
    for test_images, test_labels in test_loader:
        test_images = test_images.to(device)
        test_labels = test_labels.to(device)

        test_outputs = model(test_images)
        _, test_predicted = torch.max(test_outputs.data, 1)

        test_total += test_labels.size(0)
        test_correct += (test_predicted == test_labels).sum().item()

test_accuracy = 100 * test_correct / test_total
print(f"Accuracy of the network on the 10000 test images: {test_accuracy:.2f}%")

os.makedirs("models", exist_ok=True)
torch.save(model.state_dict(), MODEL_PATH)

print(f"Saved model to {MODEL_PATH}")
