import numpy as np
import random
import sys

import torch
import torchvision
import torchvision.transforms as transforms

from sklearn.preprocessing import OneHotEncoder
from features import get_features, get_simple_features

# Parameters
experiment = 'CIFAR10'
reg = 1
patch_shape = (6,6)
n_features = 2048
block = 200
pool_size = 3

if experiment == 'MNIST':
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])

    trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                            download=True, transform=transform)
    testset = torchvision.datasets.MNIST(root='./data', train=False,
                                           download=True, transform=transform)
    X_train = trainset.train_data.cpu().detach().numpy()
    y_train = trainset.train_labels.cpu().detach().numpy()
    X_test = testset.test_data.cpu().detach().numpy()
    y_test = testset.test_labels.cpu().detach().numpy()
elif experiment == 'CIFAR10':
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
    X_train = trainset.train_data
    y_train = np.array(trainset.train_labels)
    X_test = testset.test_data
    y_test = np.array(testset.test_labels)
    patch_shape = (6,6,3)
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

X_feat_train, X_feat_test = get_features(X_train, X_test, img_shape, n_features, block, patch_shape, pool_size)

# X_feat_train, X_feat_test = get_simple_features(X_train, X_test, 1024) #.9758 acc when 4096 features

# X_feat_train = np.loadtxt('lift_train.csv', delimiter=",")
# X_feat_test = np.loadtxt('lift_test.csv', delimiter=",")

print("Getting Matrix")
A = X_feat_train
# right = np.zeros((A.shape[1], y_train_ohe.shape[1]))
# left = np.zeros((A.shape[1], A.shape[1]))
# for i in range(A.shape[0]):
#     if i % 1000 == 0:
#         print(i)
#     right += np.outer(A[i], np.transpose(y_train_ohe[i]))
#     left += np.outer(A[i], np.transpose(A[i]))
# left = left + reg*np.identity(A.shape[1])
# w = np.dot(np.linalg.inv(left), right)
w = np.dot(np.linalg.inv(np.dot(A.T, A) + reg*np.identity(A.shape[1])), np.dot(A.T, y_train_ohe))
# w = np.dot(np.dot(A.T, np.linalg.inv(np.dot(A, A.T) + reg*np.identity(len(A)))), y_train_ohe)
print(w.shape)
print("Predicting")
y_pred = np.array([np.argmax(np.dot(np.transpose(w), x)) for x in X_feat_test])
acc =[1 if y_pred[i] == y_test[i] else 0 for i in range(len(y_pred))]
acc = sum(acc)/len(y_pred)
print("Accuracy is " + str(acc)) # 0.9907 pytorch features
