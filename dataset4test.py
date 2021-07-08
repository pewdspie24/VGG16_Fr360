import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset
import glob
import os
import cv2
import matplotlib.pyplot as plt

class CustomDataset(Dataset):
    def __init__(self, DATA_PATH):
        self.images = []
        self.labels = []
        self.convertlabels = []
        count = 0
        for label in glob.glob(DATA_PATH+'/*.jpg'):
            self.labels.append(label.split('/')[-1])
            self.images.append(label)
                # print(image)
            count+=1
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def __getitem__(self, idx):
        image_path = self.images[idx]
        label = self.labels[idx]
        image = cv2.imread(image_path)
        image = cv2.resize(image, (224,224))
        image = self.transform(image)
        return image, label

    def __len__(self):
        return len(self.labels)

    def getLabel(self, idx):
        return self.convertlabels[self.labels[idx]]

if __name__ == '__main__':
    DATA_PATH = "./fruits-360/test-multiple_fruits"
    abc = CustomDataset(DATA_PATH)
    it = abc.__getitem__(1)
    print(it[1])
    img = np.transpose(it[0], (1,2,0))
    plt.imshow(img)
    plt.show()