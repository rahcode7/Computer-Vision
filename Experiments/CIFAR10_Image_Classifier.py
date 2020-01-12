# %%
#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torchvision.transforms as transforms
import torchvision
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import numpy as np 
from pylab import *
import os 

print(os.getcwd())
LOG_PATH = 'Logs'
writer = SummaryWriter(LOG_PATH)
#!pip install torch
#!pip install torchvision
#%matplotlib inline

# %%
# from google.colab import drive
# drive.mount('/content/drive')

# %%
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5,),(0.5,))])

trainset = torchvision.datasets.CIFAR10(root='./data',train=True,download=True,transform=transform)
trainloader = torch.utils.data.DataLoader(trainset,batch_size=4,dataype='train',shuffle=True,num_workers=4)




testset = torchvision.datasets.CIFAR10(root='./data',train=False,download=True,transform=transform)
testloader = torch.utils.data.DataLoader(testset,batch_size=4,shuffle=True,num_workers=4)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


# %%
def matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))

def images_to_probs(net, images):
    '''
    Generates predictions and corresponding probabilities from a trained
    network and a list of images
    '''
    output = net(images)
    # convert output probabilities to predicted class
    _, preds_tensor = torch.max(output, 1)
    preds = np.squeeze(preds_tensor.numpy())
    return preds, [F.softmax(el, dim=0)[i].item() for i, el in zip(preds, output)]


def plot_classes_preds(net, images, labels):
    '''
    Generates matplotlib Figure using a trained network, along with images
    and labels from a batch, that shows the network's top prediction along
    with its probability, alongside the actual label, coloring this
    information based on whether the prediction was correct or not.
    Uses the "images_to_probs" function.
    '''
    preds, probs = images_to_probs(net, images)
    # plot the images in the batch, along with predicted and true labels
    fig = plt.figure(figsize=(12, 48))
    for idx in np.arange(4):
        ax = fig.add_subplot(1, 4, idx+1, xticks=[], yticks=[])
        matplotlib_imshow(images[idx], one_channel=True)
        ax.set_title("{0}, {1:.1f}%\n(label: {2})".format(
            classes[preds[idx]],
            probs[idx] * 100.0,
            classes[labels[idx]]),
                    color=("green" if preds[idx]==labels[idx].item() else "red"))
    return fig


# Define NN
from model import Net
net = Net()


# %%
import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# %%
EPOCHS = 4

 
for epoch in range(EPOCHS):
    print("Epoch {}".format(epoch))
    # For each batch of images

    for phase in ['train','val']:
        if phase == 'train':
            scheduler.step()
            model.train(True)
        else:
            model.train(False)

        running_loss = 0.0 

        for i, data  in dataloaders[phase]:

            inputs,labels = data  
            #print(inputs.shape)
            #print(labels.shape)

            optimizer.zero_grad()

            # Forward pass, backward pass is inherent
            outputs = net(inputs)

            # Calculate loss
            loss = criterion(outputs,labels)
            
            # Computer gradients of loss w.r.t weights 
            # x.grad += dloss/dx
            loss.backward()

            # Take one optimization step - Update x using x.grad 
            # x = -lr * x.grad
            optimizer.step()

            # Accumulate loss per batch of images
            running_loss += loss.item()
            #print(running_loss)
            # Print average loss per 2000 images 
            if i % 1000 == 999:    # every 1000 mini-batches...
                #print("loss for epoch " + str(epoch) + ":  " + str(running_loss))
                writer.add_scalar('Train/Loss', loss.item(), epoch)
                #writer.add_scalar('Train/Accuracy', accuracy, epoch)
            #writer.add_scalar('Accuracy/train', np.random.random(), n_iter)
                writer.add_figure('predictions vs. actuals',
                                plot_classes_preds(net, inputs, labels),
                                global_step=epoch * len(trainloader) + i)

                running_loss = 0.0
## Loss logging

writer.flush()
writer.close()


# %%
import os 
os.getcwd()

import torch 
from model import Net
MODEL_PATH = 'model_cifar10.pth'
OPTIM_PATH = 'optim_cifar10.pt'

# Saving model parameters
torch.save(net.state_dict(),MODEL_PATH)
torch.save(optimizer.state_dict(),OPTIM_PATH)



# %%
for param in net.state_dict():
    print(param,net.state_dict()[param].size())

# for param in loaded_optim.state_dict():
#     print(param,loaded_optim.state_dict()[param].size())

