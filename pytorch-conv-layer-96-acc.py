import numpy as np # linear algebra
# import lib 
import os
from tqdm.notebook import tqdm
import torch
import torchvision.transforms as transforms
import torch.optim as optim
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.utils.data import random_split
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report
from habana_frameworks.torch.utils.library_loader import load_habana_module
load_habana_module()
import habana_frameworks.torch.core as htcore



# In[2]:


# Checking device: cuda or cpu
device = torch.device("hpu")



# In[3]:


data_dir_train = 'Fruit-Images-Dataset/data/Training/'
data_dir_test = 'Fruit-Images-Dataset/data/Validation/'


# In[4]:


size_pictures = 100
data_transforms = transforms.Compose([transforms.Resize((size_pictures, size_pictures)),
                                 transforms.ToTensor()])


# In[5]:


dataset = ImageFolder(root=data_dir_train,transform=data_transforms)
data_test = ImageFolder(root=data_dir_test,transform=data_transforms)
print(len(dataset), len(data_test))


# In[12]:


train_size = int(0.8 * len(dataset)) 
val_size = len(dataset) - train_size
train_ds, val_ds = random_split(dataset, [train_size, val_size])
len(train_ds), len(val_ds) # train_ds length = dataset length - val_ds length


# In[13]:


class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=7, kernel_size=3, padding=1)
        self.act1 = nn.ReLU()
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv2d(in_channels=7, out_channels=10, kernel_size=5, padding=2)
        self.act2 = nn.ReLU()
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        
        #6250
        self.fc1 = nn.Linear(6250,  1048)
        self.act3 = nn.ReLU()
        
        self.fc2 = nn.Linear(1048, 262)
        self.act4 = nn.Sigmoid()
        
        self.fc3 = nn.Linear(262, 131)
        
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.act1(x)
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = self.act2(x)
        x = self.pool2(x)  
        
        
        x = x.view(x.size(0), x.size(1) * x.size(2) * x.size(3))
        x = self.fc1(x)
        x = self.act3(x)
        
        x = self.fc2(x)
        x = self.act4(x)
        
        x = self.fc3(x)
        return x


# In[14]:


model = NeuralNetwork()
model = model.to(device)
loss = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),  lr = 10 **(-3))
epoch = 10
batch_size = 256


# In[15]:


acc = []
def training(model, batch_size, epochs, loss, optimizer):
    for epoch in range(1, epochs + 1):
        model.train()
        dataloader = DataLoader(dataset=train_ds, batch_size=batch_size, shuffle=True)
        for (X_batch,y_batch) in tqdm(dataloader): 
            optimizer.zero_grad()
            
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            preds = model.forward(X_batch)
            
            loss_value = loss(preds, y_batch.long())
            loss_value.backward()
            
            optimizer.step()

        dataloader_test = DataLoader(dataset=val_ds, batch_size=256)
        model.eval()
        with torch.no_grad():
            summa = 0
            for (X_batch,y_batch) in tqdm(dataloader_test):
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                preds = model.forward(X_batch)
                preds = torch.max(F.softmax(preds, dim=1), dim=1)
                correct= torch.eq(preds[1], y_batch)
                summa += torch.sum(correct).item()

            acc.append(summa / len(val_ds))
            print(f'epoch: {epoch}, acc:{acc[-1]:.2%}')


# In[16]:


training(model, batch_size, epoch, loss, optimizer)


# In[18]:


dataloader_test = DataLoader(dataset=data_test, batch_size=256)
summa = 0
y_true = []
y_pred = []
for (X_batch,y_batch) in tqdm(dataloader_test):
    X_batch = X_batch.to(device)
    y_batch = y_batch.to(device)
    preds = model.forward(X_batch)
    val_predict_class = preds.argmax(dim=-1)

    y_pred.extend([predict_class.item() for predict_class in val_predict_class])
    y_true.extend([val_label.item() for val_label in y_batch])


# In[19]:


print(f'acc:{accuracy_score(y_true, y_pred):.2%}')


# In[20]:


cl_report = classification_report(y_true, y_pred,output_dict=True,
                               target_names=dataset.classes)

