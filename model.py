from matplotlib.pyplot import imshow
import torch.nn as nn
import torch
import cv2
from torchvision import transforms
from torchsummary import summary

def DoubleConv2D(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, kernel_size=3, padding=1),
        nn.BatchNorm2d(oup),
        nn.LeakyReLU(0.2),
        nn.Conv2d(oup, oup, kernel_size=3, padding=1),
        nn.BatchNorm2d(oup),
        nn.LeakyReLU(0.2),
        nn.MaxPool2d(2)
    )

def TripleFCN(inp, oup):
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(inp, oup),
        nn.BatchNorm1d(oup),
        nn.LeakyReLU(0.2),
        nn.Linear(oup, oup),
        nn.BatchNorm1d(oup),
        nn.LeakyReLU(0.2),
        
    )

class VGG16(nn.Module):
    
    def __init__(self, n_channels, n_classes):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        #Feature Extractor
        self.block1 = DoubleConv2D(n_channels, 64)
        self.block2 = DoubleConv2D(64, 128)
        self.block3 = DoubleConv2D(128, 256)
        self.block4 = DoubleConv2D(256, 512)
        self.block5 = DoubleConv2D(512, 512)
        #FCN
        self.block6 = TripleFCN(7*7*512, 4096)
        #Classifier
        self.classifier = nn.Linear(4096, n_classes)
        
    
    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        logits = self.classifier(x)
        return logits

if __name__ == "__main__":
    model = VGG16(3, 131)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    summary(model, input_size=(3, 224, 224))