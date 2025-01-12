from torch import nn
import torch
import pytorch_lightning as pl


# class Model(nn.Module):
#     """Model"""

#     def __init__(self) -> None:
#         super().__init__()
#         self.conv1 = nn.Conv2d(1, 32, 3, 1)
#         self.conv2 = nn.Conv2d(32, 64, 3, 1)
#         self.conv3 = nn.Conv2d(64, 128, 3, 1)
#         self.maxpool = nn.MaxPool2d(2, 2)
#         self.dropout = nn.Dropout(0.5)
#         self.fc = nn.Linear(128, 10)
#         self.relu = nn.ReLU()

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         x = self.relu(self.conv1(x))
#         x = self.maxpool(x)
#         x = self.relu(self.conv2(x))
#         x = self.maxpool(x)
#         x = self.relu(self.conv3(x))
#         x = self.maxpool(x)
#         x = x.view(-1, 128)
#         x = self.dropout(x)
#         x = self.fc(x)
#         return x

class Model(pl.LightningModule):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.conv3 = nn.Conv2d(64, 128, 3, 1)
        self.maxpool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(128, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.maxpool(x)
        x = self.relu(self.conv2(x))
        x = self.maxpool(x)
        x = self.relu(self.conv3(x))
        x = self.maxpool(x)
        x = x.view(-1, 128)
        x = self.dropout(x)
        x = self.fc(x)
        return x

    def criterion(self, preds, target):
        return nn.CrossEntropyLoss()(preds, target)
    
    def training_step(self, batch, batch_idx):
        data, target = batch
        preds = self(data)
        loss = self.criterion(preds, target)
        acc = (target == preds.argmax(dim=-1)).float().mean()
        self.log('train_loss', loss)
        self.log('train_acc', acc)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

    def validation_step(self, batch) -> None:
        data, target = batch
        preds = self(data)
        loss = self.criterion(preds, target)
        acc = (target == preds.argmax(dim=-1)).float().mean()
        self.log('val_loss', loss, on_epoch=True)
        self.log('val_acc', acc, on_epoch=True)

if __name__ == "__main__":
    model = Model()
    print(model)

    dummy_input = torch.randn(1, 1, 28, 28)
    output = model(dummy_input)
    print(output.shape)
