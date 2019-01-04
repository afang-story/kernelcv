import numpy as np
import scipy
import random
import sys

import torch
import torchvision
import torchvision.transforms as transforms

from sklearn.preprocessing import OneHotEncoder, scale
from features import get_features, get_simple_features, get_features_repeat, ZCA, patchify

# Parameters
experiment = 'VOC'
reg = 1
n_features = 1024*8
block = 400
pool_size = 3

if experiment == 'MNIST':
    patch_shape = (6,6)

    trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                            download=True, transform=None)
    testset = torchvision.datasets.MNIST(root='./data', train=False,
                                           download=True, transform=None)
    X_train = trainset.train_data.cpu().detach().numpy()
    y_train = trainset.train_labels.cpu().detach().numpy()
    X_test = testset.test_data.cpu().detach().numpy()
    y_test = testset.test_labels.cpu().detach().numpy()
elif experiment == 'CIFAR10':
    patch_shape = (6,6,3)

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=None)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=None)
    X_train = trainset.train_data
    y_train = np.array(trainset.train_labels)
    X_test = testset.test_data
    y_test = np.array(testset.test_labels)
elif experiment == 'VOC':
    patch_shape = (6,6,3)

    trainset = torchvision.datasets.VOCDetection(root='./data', train=True,
                                        download=True, transform=None)
    testset = torchvision.datasets.VOCDetection(root='./data', train=False,
                                       download=True, transform=None)
    X_train = trainset.train_data
    y_train = np.array(trainset.train_labels)
    X_test = testset.test_data
    y_test = np.array(testset.test_labels)
else:
    print("Not supported")
    sys.exit()

X_train = X_train / 255.
X_test = X_test / 255.

enc = OneHotEncoder(sparse=False)
y_train_ohe = enc.fit_transform(y_train.reshape(-1,1))
y_test_ohe = enc.fit_transform(y_test.reshape(-1,1))

img_shape = X_train[0].shape

X_train = X_train.reshape((len(X_train), -1))
X_test = X_test.reshape((len(X_test), -1))

# X_feat_train, X_feat_test = get_features(X_train, X_test, img_shape, n_features, block, patch_shape, pool_size)
# X_feat_train = np.float64(X_feat_train)
# X_feat_test = np.float64(X_feat_test)
#
# A = X_feat_train

# ATA = np.dot(A.T, A)
# b = np.dot(A.T, y_train_ohe)
numTrain = len(X_train)
numTest = len(X_test)
AAT = np.zeros((numTrain, numTrain), dtype='float64')
test_XT = np.zeros((numTest, numTrain), dtype='float64')
its = int(n_features / 1024)

if len(patch_shape) == 2:
    patch_shape = np.r_[patch_shape, 1]
if len(img_shape) == 2:
    img_shape = np.r_[img_shape, 1]

print('Get Patches')
X_train_c = X_train.copy()
X_test_c = X_test.copy()
patches_train = np.array([patchify(x, patch_shape, img_shape) for x in X_train_c])

print("Whiten")
patches = patches_train.reshape(-1, int(np.prod(patch_shape)))
whitener = ZCA(patches.T)
patches_train = np.dot(np.dot(patches, whitener), whitener.T).reshape(patches_train.shape)

indices = np.random.choice(range(len(patches_train.reshape(-1, int(np.prod(patch_shape))))), n_features, replace=False)
for i in range(its):
    print(i)
    X_batch_train, X_batch_test = get_features_repeat(X_train, X_test, img_shape, 1024, block, patch_shape, pool_size, patches, indices[i*1024: (i+1)*1024])
    X_batch_train = np.float64(X_batch_train)
    X_batch_test = np.float64(X_batch_test)
    AAT += np.dot(X_batch_train, X_batch_train.T)
    test_XT += np.dot(X_batch_test, X_batch_train.T)

print("Getting Matrix")
regs = [1, 10, 100, 500, 1000, 10000]
for reg in regs:
    print(reg)
    # w = scipy.linalg.solve(ATA + reg*np.identity(A.shape[1]), b, sym_pos=True)
    # w = np.dot(np.dot(A.T, np.linalg.inv(AAT + reg*np.identity(len(A)))), y_train_ohe)
    w = scipy.linalg.solve(AAT + reg*np.identity(A.shape[1]), y_train_ohe, sym_pos=True)
    print("Predicting")
    # y_pred = np.array([np.argmax(np.dot(np.transpose(w), x)) for x in X_feat_train])
    y_pred = np.array([np.argmax(np.dot(np.transpose(w), x)) for x in AAT])
    train_acc =[1 if y_pred[i] == y_train[i] else 0 for i in range(len(y_pred))]
    train_acc = sum(train_acc)/len(y_pred)
    print("Training Accuracy is " + str(train_acc))
    # y_pred = np.array([np.argmax(np.dot(np.transpose(w), x)) for x in X_feat_test])
    y_pred = np.array([np.argmax(np.dot(np.transpose(w), x)) for x in test_XT])
    acc =[1 if y_pred[i] == y_test[i] else 0 for i in range(len(y_pred))]
    acc = sum(acc)/len(y_pred)
    print("Test Accuracy is " + str(acc))
