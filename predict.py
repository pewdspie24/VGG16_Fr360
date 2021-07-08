import torch
from torch.utils.data import DataLoader
from dataset4test import CustomDataset
from model import VGG16
import tqdm as tq
import torchvision.utils as vutils
import streamlit as st

device = 'cuda' if torch.cuda.is_available() else 'cpu'
testDataset = CustomDataset('./fruits-360/test-multiple_fruits/')
testLoader = DataLoader(testDataset, batch_size = 1)
# label = testDataset.images_name
model = VGG16(3, 131).to(device)
model.load_state_dict(torch.load("./checkpointA.pth"))

with open("./labels.txt", "r") as f:
    labelList = f.readlines()

model.eval()
with open("./cc.txt", "a") as f:
    for image in tq.tqdm(testLoader):
        input = image[0].to(device)
        name = str(image[1])
        # print("")
        # print(smt)
        id = str(torch.argmin(abs(model(input)-1), dim=1)).split('[')[1].split(']')[0]
        f.write(name + ": " + id + " " + labelList[int(id)] + '\n')