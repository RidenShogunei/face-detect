import os
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from torch import nn
from tqdm import tqdm

from resnet import ResNetFaceModel
from resnetdataloader import LFWDataset
import matplotlib.pyplot as plt


def main():
    # 初始化模型
    model = ResNetFaceModel(num_classes=1)
    model.load_state_dict(torch.load('model_41.pt'))
    # 如果有GPU，使用GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # 指定数据的路径
    data_dir = 'D:\\BaiduNetdiskDownload\\lfw'
    batch_size = 80

    # 创建数据集和 DataLoader
    dataset = LFWDataset(data_dir)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    # 设置损失函数和优化器
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # 构造存储损失的空列表
    losses = []

    # 对于每一轮训练epoch
    for epoch in range(100):  # 例如我们训练10个epoch
        print(f"Training epoch {epoch + 1}")
        running_loss = 0.0
        # 对于训练集中的每一个batch
        progress_bar = tqdm(enumerate(train_loader, 0), total=len(train_loader))

        for i, data in progress_bar:
            img1, img2, labels = data[0].to(device), data[1].to(device), data[2].to(device)
            # 清空梯度
            optimizer.zero_grad()
            # 前向传播
            outputs1 = model(img1)
            outputs2 = model(img2)
            out = torch.abs(outputs1 - outputs2)
            # 计算损失
            loss = criterion(out, labels)
            # 反向传播和优化
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            # 更新进度条
            progress_bar.set_description(f"Epoch {epoch + 1} Iter {i + 1}: loss {running_loss:.5f}.")

        print('Finished Training')
        avg_loss = running_loss / len(train_loader)
        losses.append(avg_loss)
        # 在每个epoch后，保存模型参数
        if epoch % 10 == 0:
            torch.save(model.state_dict(), f'model_{epoch + 1}.pt')

    plt.plot(range(len(losses)), losses)
    plt.title("Training Loss Plot")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    # 保存图片
    plt.savefig("training_loss_plot.png")
    plt.show()

if __name__ == '__main__':
    main()
