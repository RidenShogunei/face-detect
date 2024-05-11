import torch
from module import SimpleCNN


def test():
    model = SimpleCNN()
    model.load_state_dict(torch.load('./model.ckpt'))

    test_images = ...  # 需要自己指定测试图像

    model.eval()
    outputs = model(test_images)
    _, predicted = torch.max(outputs.data, 1)

    return predicted