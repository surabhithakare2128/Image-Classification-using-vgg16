import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import json
import torchvision.models as models
from PIL import Image
import numpy as np
import argparse

#loading data function
def load_data(data_directory):
    data_directory = "./flowers/"
    data_transforms = transforms.Compose([transforms.Resize(224),
                                      transforms.RandomResizedCrop(224),
                                      transforms.RandomVerticalFlip(p=0.4),
                                      transforms.RandomHorizontalFlip(p=0.3),
                                      transforms.RandomRotation(20),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], 
                                                           [0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], 
                                                           [0.229, 0.224, 0.225])])

    validation_transforms = transforms.Compose([transforms.Resize(256),
                                            transforms.CenterCrop(224),
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.485, 0.456, 0.406], 
                                                                 [0.229, 0.224, 0.225])])

    # TODO: Load the datasets with ImageFolder
    image_datasets = datasets.ImageFolder(data_directory+'train', transform=data_transforms)
    test_datasets = datasets.ImageFolder(data_directory+'test', transform=test_transforms)
    validation_datasets = datasets.ImageFolder(data_directory+'valid', transform=validation_transforms)

    # TODO: Using the image datasets and the trainforms, define the dataloaders
    dataloaders = torch.utils.data.DataLoader(image_datasets, batch_size=64, shuffle=True)
    testloaders = torch.utils.data.DataLoader(test_datasets, batch_size = 64, shuffle = False)
    validationloaders = torch.utils.data.DataLoader(validation_datasets, batch_size=64,shuffle = False)
    
    return image_datasets,test_datasets,validation_datasets,dataloaders,testloaders,validationloaders

#building model function
def build_model(arch,hidden_units,dropout,learning_rate):
    if arch == 'vgg16':
        model = models.vgg16(pretrained=True)        
    elif arch == 'vgg13':
        model = models.vgg13(pretrained=True)
    else:
        print("{} is not a valid architecture. Valid architectures are vgg16 and vgg13. Please select one".format(arch))
    
    for param in model.parameters():
        param.requires_grad = False
    
    model.classifier = nn.Sequential(nn.Linear(model.classifier[0].in_features, hidden_units),
                                     nn.ReLU(),
                                     nn.Dropout(dropout),
                                     nn.Linear(hidden_units, 102),
                                     nn.LogSoftmax(dim=1))
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), learning_rate)
    
    return model,criterion,optimizer

#training model function
def train_model(model,criterion,optimizer, num_epochs, dataloaders, validationloaders, gpu):
    steps = 0
    running_loss = 0
    print_every = 5

    for epoch in range(num_epochs):
        for inputs, labels in dataloaders:
            steps += 1
            
            if torch.cuda.is_available() and gpu=='gpu':
                model.to('cuda')
                inputs, labels = inputs.to('cuda'), labels.to('cuda')
            
            optimizer.zero_grad()
        
            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        
            if steps % print_every == 0:
                test_loss = 0
                accuracy = 0
                model.eval()
            
            with torch.no_grad():
                test_loss = 0
                accuracy = 0
                for inputs, labels in validationloaders:
                    inputs, labels = inputs.to('cuda:0'), labels.to('cuda:0')
                    model.to('cuda:0')
                    logps = model.forward(inputs)
                    batch_loss = criterion(logps, labels)
                    
                    test_loss += batch_loss.item()
                    
                    # Calculate accuracy
                    ps = torch.exp(logps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                    
            print(f"Epoch {epoch+1}/{num_epochs}.. "
                  f"Train loss: {running_loss/print_every:.3f}.. "
                  f"Test loss: {test_loss/len(validationloaders):.3f}.. "
                  f"Test accuracy: {accuracy/len(validationloaders):.3f}")
            running_loss = 0
            model.train()
            
    print('Training Finished')
    
#saving model function
def save_model(model, image_datasets, save_directory,arch,hidden_units,dropout,learning_rate,num_epochs):
    model.class_to_idx = image_datasets.class_to_idx
    model.cpu

    checkpoint = {'architecture' :arch,
                  'input_size': model.classifier[0].in_features,
                  'output_size': 102,
                  'dropout': dropout,
                  'learning_rate': learning_rate,
                  'num_epochs': num_epochs,
                  'hidden_units': hidden_units,
                  'state_dict': model.state_dict(),
                  'class_to_idx':model.class_to_idx
                 }

    torch.save(checkpoint, save_directory)
    print("Model has been saved")
   
#loading checkpoint function
def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    arch = checkpoint['architecture']
    input_size = checkpoint['input_size']
    output_size = checkpoint['output_size']
    dropout = checkpoint['dropout']
    learning_rate = checkpoint['learning_rate']
    hidden_units = checkpoint['hidden_units']
    
    model,criterion,optimizer = build_model(arch,hidden_units,dropout,learning_rate)   
    
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    return model

#function for prediction
def predict(image_path, model, topk,gpu):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    if torch.cuda.is_available() and gpu=='gpu':
        model.to('cuda')

    pred = process_image(image_path)
    pred = pred.unsqueeze_(0)
    pred = pred.float()
    
    if gpu == 'gpu':
         pred = pred.to('cuda')
         with torch.no_grad():
            outputs = model.forward(pred)
    else:
        with torch.no_grad():
            outputs = model.forward(pred)
 
    return F.softmax(outputs.data,dim=1).topk(topk)

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    im = Image.open(image)
    
    image_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                                           std=[0.229, 0.224, 0.225])])
    tensor = image_transforms(im)
    
    return tensor