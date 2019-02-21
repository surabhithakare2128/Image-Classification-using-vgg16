import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import torchvision.models as models
import argparse
import allfunctions

parser = argparse.ArgumentParser(description='This is train.py and we will use it for training models')

parser.add_argument('data_dir', nargs='*', action="store", default="./flowers/")
parser.add_argument('--gpu', dest="gp", action="store", default="gpu", type = str)
parser.add_argument('--save_dir', dest="save_dir", action="store", default="./checkpoint.pth", type = str)
parser.add_argument('--arch', dest="architecture", action="store", default="vgg16", type = str)
parser.add_argument('--learning_rate', dest="learning_rt", action="store", default = 0.01, type = float)
parser.add_argument('--hidden_units', dest="hidden_ut", action="store", default = 12595, type=int)
parser.add_argument('--epochs', dest="epoch", action="store", default = 5, type=int)
parser.add_argument('--dropout', dest = "drop", action = "store", default = 0.3)

args = parser.parse_args()

data_directory = args.data_dir
gpu = args.gp
save_directory = args.save_dir
learning_rate = args.learning_rt
hidden_units = args.hidden_ut
num_epochs = args.epoch
arch = args.architecture
dropout = args.drop

#loading data
image_datasets,test_datasets,validation_datasets,dataloaders,testloaders,validationloaders = allfunctions.load_data(data_directory)

#building model
model,criterion,optimizer = allfunctions.build_model(arch,hidden_units,dropout,learning_rate)

#training model
allfunctions.train_model(model,criterion,optimizer, num_epochs, dataloaders, validationloaders, gpu)

#saving model
allfunctions.save_model(model, image_datasets, save_directory,arch,hidden_units,dropout,learning_rate,num_epochs)