import torch.optim as optim
from torch import nn
import matplotlib.pyplot as plt

from dataloader import get_loader
from module import SimpleCNN
import torch


def train():
    data_dir = 'D:\BaiduNetdiskDownload\lfw'
    batch_size = 32
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device', device)

    model = SimpleCNN(num_classes=5749).to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.001)  # Reduced learning rate
    criterion = nn.CrossEntropyLoss()

    print("Preparing train loader.")
    train_loader = get_loader(data_dir, batch_size)
    print("Train loader prepared.")

    num_epochs = 500
    print("Training started.")
    model.train()

    # Initialize empty lists to store loss and epoch values
    losses = []
    epochs = []

    for epoch in range(num_epochs):
        print("Epoch {} started.".format(epoch + 1))
        total_loss = 0
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            labels = labels

            loss = criterion(outputs, labels)
            total_loss += loss.item()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)  # Gradient clipping

            optimizer.step()

            if (i + 1) % 100 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                      .format(epoch + 1, num_epochs, i + 1, len(train_loader), loss.item()))

        # Calculate average loss for the epoch
        avg_loss = total_loss / len(train_loader)
        losses.append(avg_loss)
        epochs.append(epoch + 1)

        print("Epoch {} ended. Average loss: {:.4f}".format(epoch + 1, avg_loss))

    torch.save(model.state_dict(), './model.ckpt')
    print("Training finished. Model saved to './model.ckpt'")

    # Plotting the loss curve
    plt.plot(epochs, losses)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.show()

train()