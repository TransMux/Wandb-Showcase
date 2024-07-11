import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import typer
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import wandb

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Net(nn.Module):
    def __init__(self, dropout: float, hidden_size: int):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d(dropout)
        self.fc1 = nn.Linear(320, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 10)
        self.dropout = dropout

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


def train(model, train_loader, optimizer, epoch, log_interval):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print(
                f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({batch_idx / len(train_loader):.0%})]\tLoss: {loss.item():.6f}"
            )


def test(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            # sum up batch loss
            test_loss += F.nll_loss(output, target, reduction="sum").item()
            # get the index of the max log-probability
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print(
        f"Test set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({correct / len(test_loader.dataset):.0%})\n"
    )
    wandb.log(
        {
            "Test Accuracy": 100.0 * correct / len(test_loader.dataset),
            "Test Loss": test_loss,
        }
    )


def main(
    batch_size: int = typer.Option(64, help="Input batch size for training"),
    test_batch_size: int = typer.Option(1000, help="Input batch size for testing"),
    epochs: int = typer.Option(10, help="Number of epochs to train"),
    lr: float = typer.Option(0.01, help="Learning rate"),
    momentum: float = typer.Option(0.5, help="SGD momentum"),
    dropout: float = typer.Option(0.5, help="Dropout rate"),
    hidden_size: int = typer.Option(10, help="Linear Layer hidden size"),
    seed: int = typer.Option(42, help="Random seed"),
    log_interval: int = typer.Option(
        10, help="How many batches to wait before logging training status"
    ),
):
    """
    Main function for training and testing the MNIST model.
    """
    wandb.init(config=locals())

    torch.manual_seed(seed)

    train_loader = DataLoader(
        datasets.MNIST(
            "../data",
            train=True,
            download=True,
            transform=transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
            ),
        ),
        batch_size=batch_size,
        shuffle=True,
        num_workers=1,
        pin_memory=True,
    )

    test_loader = DataLoader(
        datasets.MNIST(
            "../data",
            train=False,
            transform=transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
            ),
        ),
        batch_size=test_batch_size,
        shuffle=True,
        num_workers=1,
        pin_memory=True,
    )

    model = Net(dropout, hidden_size).to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    wandb.watch(model)

    for epoch in range(1, epochs + 1):
        train(model, train_loader, optimizer, epoch, log_interval)
        test(model, test_loader)


if __name__ == "__main__":
    typer.run(main)
