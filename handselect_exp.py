import numpy as np
import scipy
import random
import sys
import csv
import glob
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from scipy.special import softmax
from skimage.transform import pyramid_gaussian
from sklearn.preprocessing import OneHotEncoder, scale
from sklearn.metrics import average_precision_score
from features import get_features, get_simple_features, get_features_repeat, ZCA, patchify # visualizer
from voc_helpers.ptvoc import VOCClassification, VOCDetection
from average_precision.python.ap import compute_multiple_aps
# Parameters
experiment = 'VOC'
n_features = 1
block_n = 32
block_f = 1
pool_size = 1
if experiment == 'VOC':
    dim = 256
    transform = torchvision.transforms.Resize((dim, dim))
    yr = '2012'
    trainset = VOCClassification(root='./data', image_set='train', year=yr,
                                        download=True, transform=transform)
    flip = torchvision.transforms.Compose([transform, torchvision.transforms.RandomHorizontalFlip(1)])
    trainsetflip = VOCClassification(root='./data', image_set='train', year=yr, download=True, transform=flip)
    valset = VOCClassification(root='./data', image_set='val', year=yr,
                                       download=True, transform=transform)
    X_train = trainset.data
    y_train = np.array(trainset.labels)
    X_test = valset.data
    y_test = np.array(valset.labels)
    X_train = np.vstack((X_train, trainsetflip.data))
    y_train = np.vstack((y_train, np.array(trainsetflip.labels)))
    y_train_ohe = y_train
    y_test_ohe = y_test
X_train = X_train / 255.
X_test = X_test / 255.
img_shape = X_train[0].shape

X_train = X_train.reshape((len(X_train), -1))
X_test = X_test.reshape((len(X_test), -1))

numTrain = len(X_train)
numTest = len(X_test)
AAT = np.zeros((numTrain, numTrain), dtype='float64')
test_XT = np.zeros((numTest, numTrain), dtype='float64')
train_lift = None
test_lift = None

print("Get Patch")
# Get patch from image, find way to normalize - probably just take patch from data after centering and making unit variance, or whiten etc.
# maybe normalize all data initially
patches_train = np.zeros(40,40,3) # load image here
patch_shape = patch.shape
indices = range(1)
print("Convolve")
X_batch_train, X_batch_test = get_features_repeat(X_train, X_test, img_shape, block_f, block_n, patch_shape, pool_size, patches_train, indices)

# Get indices of images in each class, index into the values from convolve step and get stats (min, mean, med, max) 
# Then do for multiple patches of a class and average over
