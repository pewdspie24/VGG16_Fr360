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
        for label in os.listdir(DATA_PATH):
            self.convertlabels.append(label)
            for image in glob.glob(os.path.join(DATA_PATH, label) + "/*.jpg"):
                self.labels.append(count)
                self.images.append(image)
                print(image, label)
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
    DATA_PATH = "./fruits-360/Training/"
    abc = CustomDataset(DATA_PATH)
    it = abc.__getitem__(321)
    print(it[1])
    print(it.getLabel())
    img = np.transpose(it[0], (1,2,0))
    plt.imshow(img)
    plt.show()