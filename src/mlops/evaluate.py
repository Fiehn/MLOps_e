import torch
import typer
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from model import Model
from data import corrupt_mnist

app = typer.Typer()

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)


@app.command()
def evaluate(batch_size: int = 100) -> None:
    # Load data
    train_set, test_set = corrupt_mnist()
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    # Load model
    model = Model().to(DEVICE)
    model.load_state_dict(torch.load("models/model.pt", weights_only=True))
    model.eval()

    # Test the model
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        print(f"Accuracy of the model on the test images: {100 * correct / total} %")


if __name__ == "__main__":
    app()
