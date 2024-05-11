import os
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from torch import nn
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from resnet import ResNetFaceModel
from resnetdataloader import LFWDataset
import matplotlib.pyplot as plt


def main():
    writer = SummaryWriter()  # 初始化一个 SummaryWriter

    model = ResNetFaceModel(num_classes=1)
    model.load_state_dict(torch.load('model_41.pt'))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    data_dir = 'D:\\BaiduNetdiskDownload\\lfw'
    batch_size = 92

    dataset = LFWDataset(data_dir)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    losses = []

    for epoch in range(100):
        print(f"Training epoch {epoch + 1}")
        running_loss = 0.0

        progress_bar = tqdm(enumerate(train_loader, 0), total=len(train_loader))

        for i, data in progress_bar:
            img1, img2, labels = data[0].to(device), data[1].to(device), data[2].to(device)

            optimizer.zero_grad()

            outputs1 = model(img1)
            outputs2 = model(img2)
            out = torch.abs(outputs1 - outputs2)

            loss = criterion(out, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            progress_bar.set_description(f"Epoch {epoch + 1} Iter {i + 1}: loss {running_loss / len(train_loader):.5f}.")
        avg_loss = running_loss / len(train_loader)
        writer.add_scalar("Loss/train", avg_loss, epoch)  # 向tensorboard添加损失数据
        print('Finished Training')

        losses.append(avg_loss)

        if epoch % 10 == 0:
            torch.save(model.state_dict(), f'model_{epoch + 1}.pt')

    plt.plot(range(len(losses)), losses)
    plt.title("Training Loss Plot")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    plt.savefig("training_loss_plot.png")
    plt.show()

    writer.close()  # 关闭 SummaryWriter


if __name__ == '__main__':
    main()