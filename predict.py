import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import torchvision.models as models
import numpy as np
import argparse
import json
import PIL
from PIL import Image
import allfunctions

parser = argparse.ArgumentParser(description='This is predict.py and we will use it for predicting the image class')

parser.add_argument('data_dir', nargs='*', action="store", default="./flowers/")
parser.add_argument('--save_dir', dest="save_dir", action="store", default="./checkpoint.pth", type = str)
parser.add_argument('--image', action='store', default = 'flowers/train/102/image_08000.jpg', type = str)
parser.add_argument('--gpu', dest="gp", action="store", default="gpu", type = str)
parser.add_argument('--top_k', default=5, dest="top_k", action="store", type=int)

args = parser.parse_args()

data_directory = args.data_dir
save_directory = args.save_dir
img = args.image
gpu = args.gp
topk = args.top_k

image_datasets,test_datasets,validation_datasets,dataloaders,testloaders,validationloaders = allfunctions.load_data(data_directory)

model = allfunctions.load_checkpoint(save_directory)

with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)
    
#probs, classes = allfunctions.predict(img, model, topk, gpu)
predictions = allfunctions.predict(img, model, topk, gpu)
probabilities = np.array(predictions[0][0])
name = np.array(predictions[1][0])
index = 1
names = [cat_to_name[str(index + 1)] for index in name ]

for top_probability in range(0,topk):
    print("{} class has a probability of {}".format(names[top_probability], probabilities[top_probability]))