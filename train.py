import torch
from torch import nn
from torch.utils.data import DataLoader
from dataset import CustomDataset
from model import VGG16
import tqdm as tq #for pc
# import tqdm.notebook as tq #for colab
import torch.optim as optim
from matplotlib import pyplot as plt
import numpy as np

batch_size = 8
LR = 1e-3
epochs = 15
device = 'cuda' if torch.cuda.is_available() else 'cpu'

trainDataset = CustomDataset('./fruits-360/Training/')
trainLoader = DataLoader(trainDataset, batch_size=batch_size, shuffle=True, num_workers=2)
 
validDataset = CustomDataset('./fruits-360/Test/')
validLoader = DataLoader(validDataset, batch_size=batch_size, num_workers=2)

model = VGG16(3,131).to(device)

optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=0)
criterion = nn.CrossEntropyLoss()
#train
losses_train = []
losses_valid = []
for epoch in range(epochs):
    running_train_loss = 0.0
    running_valid_loss = 0.0
    
    model.train()
    for (input, label) in tq.tqdm(trainLoader): #for colab
        optimizer.zero_grad()
        input = input.to(device)
        label = label.to(device)
        output = model(input)
        loss = criterion(output, torch.max(label, 1)[1])
        loss.backward()
        optimizer.step()
        running_train_loss += loss.item() * input.size(0) 
    epoch_train_loss = running_train_loss / len(trainLoader)
    losses_train.append(epoch_train_loss)
    print("")
    print('Training, Epoch {} - Loss {}'.format(epoch+1, epoch_train_loss))

    model.eval()
    with torch.no_grad():
      for (input, label) in tq.tqdm(validLoader): #for colab
        input = input.to(device)
        label = label.to(device)
        output = model(input)
        loss = criterion(output, torch.max(label, 1)[1])
        running_valid_loss += loss.item() * input.size(0)
      epoch_valid_loss = running_valid_loss / len(validLoader)
      losses_valid.append(epoch_valid_loss)
      print("")
      print('Validating, Epoch {} - Loss {}'.format(epoch+1, epoch_valid_loss))
      
    torch.save(model.state_dict(), "./checkpointA.pth")
      
plt.plot(losses_train, label="train")
plt.plot(losses_valid, label="valid") 
plt.legend(loc="upper left")
torch.save(model.state_dict(), "./checkpointA.pth")