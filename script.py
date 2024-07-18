# %% [markdown]
# This is the fine-tuned model.

# %%
from __future__ import print_function, division

import torch
import torch.optim as optim

from torch.optim import lr_scheduler
from torchvision import datasets, transforms
import time
import os

# To avoid kernel Crash
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


import torch.nn as nn
import torch.nn.functional as F

# for plotting
import numpy as np
import matplotlib.pyplot as plt


# TODO: You can modify the network architecture
class Net(nn.Module):
    """
    Input - 1x32x32
    Output - 10
    """
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=7, padding=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(p=0.2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(p=0.2),

            nn.Flatten(), 
            nn.Linear(256*4*4, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
            )
        
    def forward(self, xb):
        return self.network(xb)

# %%
# ResNet
class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.conv = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self.make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self.make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self.make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self.make_layer(block, 512, num_blocks[3], stride=2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def make_layer(self, block, out_channels, num_blocks, stride):
        layers = []
        layers.append(block(self.in_channels, out_channels, stride))
        self.in_channels = out_channels
        for _ in range(1, num_blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.relu(self.bn(self.conv(x)))
        out = self.layer1(out)
        out = nn.Dropout(p=0.2)(out)
        out = self.layer2(out)
        out = nn.Dropout(p=0.2)(out)
        out = self.layer3(out)
        out = nn.Dropout(p=0.2)(out)
        out = self.layer4(out)
        out = nn.Dropout(p=0.2)(out) # add drop-out regularization
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

# %%
class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.conv = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self.make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self.make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self.make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self.make_layer(block, 512, num_blocks[3], stride=2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def make_layer(self, block, out_channels, num_blocks, stride):
        layers = []
        layers.append(block(self.in_channels, out_channels, stride))
        self.in_channels = out_channels
        for _ in range(1, num_blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.relu(self.bn(self.conv(x)))
        out = self.layer1(out)
        out = nn.Dropout(p=0.2)(out)
        out = self.layer2(out)
        out = nn.Dropout(p=0.2)(out)
        out = self.layer3(out)
        out = nn.Dropout(p=0.2)(out)
        out = self.layer4(out)
        out = nn.Dropout(p=0.2)(out) # add drop-out regularization
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

# %%
# TODO: You can try different augmentation strategies

data_transforms_1 = {
    'train': transforms.Compose([
        transforms.RandomAffine(degrees=10, translate=(0,0.1)),
        transforms.Resize((32,32)),
        transforms.RandomResizedCrop(size=32, scale=(0.8, 0.95)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),

    ]),
    'test': transforms.Compose([
        transforms.Resize((32,32)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
}

# %%
data_dir = 'data' # Suppose the dataset is stored under this folder
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms_1[x])
                  for x in ['train', 'test']} # Read train and test sets, respectively.

train_dataloader = torch.utils.data.DataLoader(image_datasets['train'], batch_size=128,
                                             shuffle=True, num_workers=4)

test_dataloader = torch.utils.data.DataLoader(image_datasets['test'], batch_size=128,
                                             shuffle=False, num_workers=4)

train_size =len(image_datasets['train'])

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # Set device to "cpu" if you have no gpu
print(torch.cuda.is_available())
print(device)

# %%
def train_test(model, criterion, optimizer, scheduler, num_epochs=25):
    train_loss = []
    train_accuracy = []
    val_loss = [] 
    val_accuracy = []
    history = dict()
    model.train()
    for epoch in range(num_epochs):
        running_training_loss = 0.0
        running_training_accuracy = 0.0
        iteration_training_loss = 0.0
        total_training_predictions = 0
       
        start_time = time.time()
        for i, data in enumerate(train_dataloader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_training_loss += loss.item()*inputs.size(0)
            
            _, predicted = torch.max(outputs.data, 1)
            total_training_predictions += labels.size(0)
            running_training_accuracy += (predicted == labels).sum().item()
            iteration_training_loss += loss.item() 
            if (i+1) % 100 == 0:
                print('Epoch:[%d]-Iteration:[%d], training loss: %.3f' %
                      (epoch + 1,i+1,iteration_training_loss/(i+1)))
        end_time = time.time()
        print('Time cost of one epoch: [%d]s' % (end_time-start_time))
        
        epoch_training_accuracy = running_training_accuracy / train_size*100
        epoch_training_loss = running_training_loss / train_size
        
        print('Epoch:[%d], training accuracy: %.1f, training loss: %.3f' %
              (epoch + 1,epoch_training_accuracy, epoch_training_loss))
        
        train_loss.append(epoch_training_loss)
        train_accuracy.append(epoch_training_accuracy)
        
        scheduler.step(epoch_training_loss)
        
    print('Finished Training')

    history['train_loss'] = train_loss
    history['train_accuracy'] = train_accuracy

    correct = 0
    total = 0
    Catagorical_acc = {i:0 for i in range(10)}
    model.eval()

    # Since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in test_dataloader:
            images, labels = data[0].to(device), data[1].to(device)
            # calculate outputs by running images through the network
            outputs = model(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            for index,label in enumerate(labels):
                Catagorical_acc[label.item()] += (predicted[index] == label).item()
                
    accuracy = 100 * correct / total
    print('Accuracy of the network on test images: %.2f %%' % (
            accuracy))
    print("Catagorical Accuracies:")
    for lable, acc in enumerate(Catagorical_acc):
        print(f"Label {lable}: {Catagorical_acc[acc]/500 * 100 : .2f}%")
    return history, accuracy


# %%
# for plotting
def plot_graph(history:dict, accuracy:float, title:str):
    plt.plot(np.array(history['train_accuracy']))
    plt.axhline(y = accuracy, color = 'r', linestyle = '--', label = 'Test Accuracy') 
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy%')
    plt.title(title)
    plt.legend(['Train Accuracy', f'Final Test Accuracy ({accuracy})'])
    plt.show()

# %%
if __name__ == '__main__':
    end = time.time()
    resnet = ResNet(BasicBlock, [2,4,4,2]).to(device)
    
    criterion = nn.CrossEntropyLoss()
    
    optimizer = optim.Adam(resnet.parameters(), lr = 5e-4)
    
    exp_lr_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,mode = "max")
    
    history, accuracy = train_test(resnet, criterion, optimizer, exp_lr_scheduler,
            num_epochs=30)

    
    print("time required %.2fs" %(time.time() - end))
    plot_graph(history, accuracy, 'ResNet : 30 epochs')
    



