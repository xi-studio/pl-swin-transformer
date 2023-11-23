import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, Compose
import torchvision.datasets as datasets
from accelerate import Accelerator
from accelerate import notebook_launcher

train_data = datasets.FashionMNIST(
    root="./data",
    train=True,
    download=True,
    transform=Compose([ToTensor()])
)

test_data = datasets.FashionMNIST(
    root="./data",
    train=False,
    download=True,
    transform=Compose([ToTensor()])
)


class CNNModel(nn.Module):
    
  def __init__(self):
    super(CNNModel, self).__init__()
    self.module1 = nn.Sequential(
        nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2)
    )  
    self.module2 = nn.Sequential(
        nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2)
    )
    self.flatten = nn.Flatten()
    self.linear1 = nn.Linear(7 * 7 * 64, 64)
    self.linear2 = nn.Linear(64, 10)
    self.relu = nn.ReLU()
    
  def forward(self, x):
    out = self.module1(x)
    out = self.module2(out)
    out = self.flatten(out)
    out = self.linear1(out)
    out = self.relu(out)
    out = self.linear2(out)
    return out

def training_function():
    epoch_num = 4
    batch_size = 64
    learning_rate = 0.005

    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

    model = CNNModel()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    accelerator = Accelerator()
    model, optimizer, train_loader, val_loader = accelerator.prepare(model, optimizer, train_loader, val_loader)

    for epoch in range(epoch_num):
        model.train()
        for i, (X_train, y_train) in enumerate(train_loader):
            out = model(X_train)
            loss = criterion(out, y_train)

            optimizer.zero_grad()
            accelerator.backward(loss)
            optimizer.step()

            if (i + 1) % 100 == 0:
                print(f"{accelerator.device} Train... [epoch {epoch + 1}/{epoch_num}, step {i + 1}/{len(train_loader)}]\t[loss {loss.item()}]")
        

def main(): 
    training_function()

if __name__ == '__main__':
    main()
