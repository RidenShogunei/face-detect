import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from resnet import ResNetFaceModel
from dataloader import LFWDataset

import matplotlib.pyplot as plt


def main():
    writer = SummaryWriter()

    model = ResNetFaceModel()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(device)
    data_dir = 'D:\\BaiduNetdiskDownload\\lfw'
    batch_size = 64

    dataset = LFWDataset(data_dir)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    # 选择MSELoss作为损失函数
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    losses = []

    from tqdm import tqdm

    # 在外层循环之前打印开始训练的消息
    print("Start training...")

    for epoch in range(100):
        print(f"Training epoch {epoch + 1}")
        running_loss = 0.0
        progress_bar = tqdm(enumerate(train_loader, 0), total=len(train_loader))

        for i, data in progress_bar:
            img1, img2, euclidean_distance = data[0].to(device), data[1].to(device), data[2].to(device)
            optimizer.zero_grad()
            outputs1 = model(img1)
            outputs2 = model(img2)
            # 计算真实欧式距离
            real_distance = torch.sqrt(((outputs1 - outputs2) ** 2).sum(axis=1))
            # 使用MSELoss计算损失
            loss = criterion(euclidean_distance, real_distance)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            progress_bar.set_description(f"Epoch {epoch + 1} Iter {i + 1}")
            progress_bar.set_postfix(loss=f"{running_loss / (i + 1):.5f}")
        avg_loss = running_loss / len(train_loader)
        writer.add_scalar("Loss/train", avg_loss, epoch)
        print('Finished Training')
        losses.append(avg_loss)

        if epoch % 10 == 0:
            torch.save(model.state_dict(), f'model_{epoch + 1}.pt')

    # 训练完成后打印结束训练的消息
    print("Training completed.")

    plt.plot(range(len(losses)), losses)
    plt.title("Training Loss Plot")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig("training_loss_plot.png")
    plt.show()

    writer.close()


if __name__ == '__main__':
    main()