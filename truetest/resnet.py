import torch.nn as nn
import torchvision.models as models

class ResNetFaceModel(nn.Module):
    def __init__(self, num_classes=1):
        super(ResNetFaceModel, self).__init__()

        # Load pre-trained ResNet50
        self.resnet = models.resnet50(pretrained=True)

        # Remove the last layer
        modules = list(self.resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)

        # Add new layers
        self.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        x = self.resnet(x)
        x = x.view(x.size(0), -1)  # flatten the tensor
        x = self.fc(x)

        return x