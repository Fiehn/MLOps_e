import torch
import typer
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from data import lightning_data
from model import Model
import wandb
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

app = typer.Typer()

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

# @app.command()
# def train(
#     learning_rate: float = typer.Option(0.001, "--learning_rate", "-lr", help="Learning rate for training"),
#     epochs: int = typer.Option(10, "--epochs", "-e", help="Number of epochs for training"),
#     batch_size: int = typer.Option(100, "--batch_size", "-bs", help="Batch size for training"),
# ) -> None:
#     lr = learning_rate
#     # init wandb
#     wandb.init(project="corrupt-mnist", config={"lr": lr, "epochs": epochs, "batch_size": batch_size})
    
#     # Load data
#     train_set, test_set = corrupt_mnist()
#     train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

#     # Load model
#     model = Model().to(DEVICE)

#     # Loss and optimizer
#     criterion = torch.nn.CrossEntropyLoss()
#     optimizer = torch.optim.Adam(model.parameters(), lr=lr)

#     statics = {"train_loss": [], "train_acc": []}
#     # Train the model
#     total_step = len(train_loader)
#     for epoch in range(epochs):
#         model.train()
#         total_loss = 0
#         preds, targets = [], []
#         for i, (images, labels) in enumerate(train_loader):
#             # Forward pass
#             outputs = model(images)
#             loss = criterion(outputs, labels)
#             # Backward and optimize
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#             statics["train_loss"].append(loss.item())
#             total_loss += loss.item()
#             # Track the accuracy
#             accuracy = (outputs.argmax(dim=1) == labels).float().mean()
#             statics["train_acc"].append(accuracy.item())

#             if (i + 1) % 100 == 0:
#                 print(
#                     f"Epoch [{epoch+1}/{epochs}], Step [{i+1}/{total_step}], Loss: {loss.item()}"
#                 )
#                 # add a plot of the input images
#                 images_collection = wandb.Image(images[:5].detach().cpu(), caption="Input images")
#                 wandb.log({"images": images_collection})

#                 # add a plot of histogram of the gradients
#                 grads = torch.cat([p.grad.flatten() for p in model.parameters() if p.grad is not None], 0)
#                 wandb.log({"gradients": wandb.Histogram(grads)})
#         avg_loss = total_loss / (i + 1)
#         wandb.log({"epoch": epoch, "loss": avg_loss})
#     # Test the model
    
#     print("Training Completed")
#     fig, axs = plt.subplots(1, 2, figsize=(15, 5))
#     axs[0].plot(statics["train_loss"])
#     axs[0].set_title("Train loss")
#     axs[1].plot(statics["train_acc"])
#     axs[1].set_title("Train accuracy")
#     wandb.log({"training_stats": wandb.Image("reports/training_stats1.png")})
#     # Save the model

#     final_accuracy = statics["train_acc"][-1]
#     torch.save(model.state_dict(), "models/model.pt")
#     artifact = wandb.Artifact(
#         name="corrupt_mnist_model",
#         type="model",
#         description="A model trained to classify corrupt MNIST images",
#         metadata={"accuracy": final_accuracy},
#     )
#     artifact.add_file("models/model.pt")
#     wandb.log_artifact(artifact)

@app.command()
def train(
    learning_rate: float = typer.Option(0.001, "--learning_rate", "-lr", help="Learning rate for training"),
    epochs: int = typer.Option(10, "--epochs", "-e", help="Number of epochs for training"),
    batch_size: int = typer.Option(100, "--batch_size", "-bs", help="Batch size for training"),
) -> None:
    
    logger = WandbLogger(project="corrupt-mnist", config={"lr": learning_rate, "epochs": epochs, "batch_size": batch_size})

    train_loader, test_loader = lightning_data(batch_size)
    model = Model()
    
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath="models", filename="best-checkpoint", monitor="val_loss", mode="min"
    )
    stopping_callback = pl.callbacks.EarlyStopping(monitor="val_loss", patience=3)
    trainer = pl.Trainer(
        max_epochs=epochs,
        logger=logger,
        callbacks=[checkpoint_callback, stopping_callback],
    )

    trainer.fit(model, train_loader, test_loader)

if __name__ == "__main__":
    app()
