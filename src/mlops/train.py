import torch
import typer
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from .data import corrupt_mnist
from .model import Model

app = typer.Typer()

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)


@app.command()
def train(lr: float = 0.001, epochs: int = 10, batch_size: int = 100) -> None:
    # Load data
    train_set, test_set = corrupt_mnist()
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

    # Load model
    model = Model().to(DEVICE)

    # Loss and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    statics = {"train_loss": [], "train_acc": []}
    # Train the model
    total_step = len(train_loader)
    for epoch in range(epochs):
        for i, (images, labels) in enumerate(train_loader):
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            statics["train_loss"].append(loss.item())

            # Track the accuracy
            accuracy = (outputs.argmax(dim=1) == labels).float().mean()
            statics["train_acc"].append(accuracy.item())

            if (i + 1) % 100 == 0:
                print(
                    f"Epoch [{epoch+1}/{epochs}], Step [{i+1}/{total_step}], Loss: {loss.item()}"
                )
    print("Training Completed")
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    axs[0].plot(statics["train_loss"])
    axs[0].set_title("Train loss")
    axs[1].plot(statics["train_acc"])
    axs[1].set_title("Train accuracy")
    fig.savefig("reports/training_stats1.png")
    # Save the model
    torch.save(model.state_dict(), "models/model.pt")
    print("Model saved")


if __name__ == "__main__":
    app()
