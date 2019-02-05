import numpy as np
import scipy
import random
import sys

import torch
import torchvision
import torchvision.transforms as transforms

from sklearn.preprocessing import OneHotEncoder, scale
from sklearn.metrics import average_precision_score
from features import get_features, get_simple_features, get_features_repeat, ZCA, patchify
from voc_helpers.ptvoc import VOCClassification
# Parameters
experiment = 'VOC'
# reg = 1
threshold = [.25, .33, .4, .5, .6, .66, .75, .8]
# threshold = [0]
n_features = 4*1024
# n_features = 512
block_f = 256 # for 256 x 256 and 6x6
block_n = 16
# block_f = 512 # for 128 x 128 and 6x6
# block_n = 32
# block_f = 1024
# block_n = 200
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
    dim = 256
    # transform = torchvision.transforms.CenterCrop(256)
    # transform = torchvision.transforms.Resize((256, 256))
    transform = torchvision.transforms.Compose([torchvision.transforms.Resize(dim), torchvision.transforms.CenterCrop(dim)])
    patch_shape = (6,6,3)
    # patch_shape2 = (12,12,3)
    yr = '2012'
    trainset = VOCClassification(root='./data', image_set='train', year=yr,
                                        download=True, transform=transform)
    flip = torchvision.transforms.Compose([torchvision.transforms.Resize(dim), torchvision.transforms.CenterCrop(dim), torchvision.transforms.RandomHorizontalFlip(1)])
    trainsetflip = VOCClassification(root='./data', image_set='train', year=yr, download=True, transform=flip)
    valset = VOCClassification(root='./data', image_set='val', year=yr,
                                       download=True, transform=transform)
    # testset = VOCClassification(root='./data', image_set='test', year=yr,
    #                                     download=True, transform=crop)
    X_train = trainset.data
    y_train = np.array(trainset.labels)
    X_test = valset.data
    y_test = np.array(valset.labels)
    '''
    X_all = np.vstack((X_train, X_test))
    y_all = np.vstack((y_train, y_test))
    sn = len(X_all)
    indices = np.random.choice(range(sn), int(.8*sn), replace=False)
    others = np.setdiff1d(range(sn), indices)
    X_train = X_all[indices]
    X_test = X_all[others]
    y_train = y_all[indices]
    y_test = y_all[others]
    '''
    X_train = np.vstack((X_train, trainsetflip.data))
    y_train = np.vstack((y_train, np.array(trainsetflip.labels)))
    y_train_ohe = y_train
    y_test_ohe = y_test
    # X_train = np.vstack((X_train, X_val))
    # y_train = np.vstack((y_train, y_val))
    # X_test = testset.data
    # y_test = np.array(testset.labels)
    print(len(X_train))
    print(len(X_test))
    # print(y_train)
else:
    print("Not supported")
    sys.exit()

X_train = X_train / 255.
X_test = X_test / 255.

# enc = OneHotEncoder(sparse=False)
# y_train_ohe = enc.fit_transform(y_train.reshape(-1,1))
# y_test_ohe = enc.fit_transform(y_test.reshape(-1,1))

img_shape = X_train[0].shape

X_train = X_train.reshape((len(X_train), -1))
X_test = X_test.reshape((len(X_test), -1))

# X_feat_train, X_feat_test = get_features(X_train, X_test, img_shape, n_features, block_n, patch_shape, pool_size)
# X_feat_train = np.float64(X_feat_train)
# X_feat_test = np.float64(X_feat_test)
# A = X_feat_train
# AAT = np.dot(A, A.T)
# test_XT = np.dot(X_feat_test, X_feat_train.T)
# ATA = np.dot(A.T, A)
# b = np.dot(A.T, y_train_ohe)

numTrain = len(X_train)
numTest = len(X_test)
AAT = np.zeros((numTrain, numTrain), dtype='float64')
test_XT = np.zeros((numTest, numTrain), dtype='float64')
its = int(n_features / block_f)

if len(patch_shape) == 2:
    patch_shape = np.r_[patch_shape, 1]
if len(img_shape) == 2:
    img_shape = np.r_[img_shape, 1]

print('Get Patches')
X_train_c = X_train.copy()
X_test_c = X_test.copy()
patches_train = np.array([patchify(x, patch_shape, img_shape)[np.random.choice((img_shape[0]-patch_shape[0]+1)**2, 100, replace=False)] for x in X_train_c])
# patches_train2 = np.array([patchify(x, patch_shape2, img_shape)[np.random.choice((img_shape[0]-patch_shape2[0]+1)**2, 100, replace=False)] for x in X_train_c])

print("Whiten")
patches = patches_train.reshape(-1, int(np.prod(patch_shape)))
whitener = ZCA(patches.T)
patches_train = np.dot(np.dot(patches, whitener), whitener.T).reshape(patches_train.shape)

# patches2 = patches_train2.reshape(-1, int(np.prod(patch_shape2)))
# whitener2 = ZCA(patches2.T)
# patches_train2 = np.dot(np.dot(patches2, whitener2), whitener2.T).reshape(patches_train2.shape)

indices = np.random.choice(range(len(patches_train.reshape(-1, int(np.prod(patch_shape))))), n_features, replace=False)
# indices2 = np.random.choice(range(len(patches_train2.reshape(-1, int(np.prod(patch_shape2))))), n_features, replace=False)

for i in range(its):
    print(i)
    X_batch_train, X_batch_test = get_features_repeat(X_train, X_test, img_shape, block_f, block_n, patch_shape, pool_size, patches_train, indices[i*block_f: (i+1)*block_f])
    X_batch_train = np.float64(X_batch_train)
    X_batch_test = np.float64(X_batch_test)
    
    # X_batch_train2, X_batch_test2 = get_features_repeat(X_train, X_test, img_shape, block_f, block_n, patch_shape2, pool_size, patches_train2, indices2[i*block_f: (i+1)*block_f])
    # X_batch_train2 = np.float64(X_batch_train2)
    # X_batch_test2 = np.float64(X_batch_test2)

    # X_batch_train = np.hstack((X_batch_train, X_batch_train2))
    # X_batch_test = np.hstack((X_batch_test, X_batch_test2))

    AAT += np.dot(X_batch_train, X_batch_train.T)
    test_XT += np.dot(X_batch_test, X_batch_train.T)
save_indices = []
save_labels = []
print("Getting Matrix")
regs = [1, 10, 100, 500, 1000, 10000, 100000, 1000000]
for reg in regs:
    print(reg)
    # w = scipy.linalg.solve(ATA + reg*np.identity(A.shape[1]), b, sym_pos=True)
    # w = np.dot(np.dot(A.T, np.linalg.inv(AAT + reg*np.identity(len(A)))), y_train_ohe)
    w = scipy.linalg.solve(AAT + reg*np.identity(AAT.shape[1]), y_train_ohe, sym_pos=True)
    print("Predicting")
    # y_pred = np.array([np.argmax(np.dot(np.transpose(w), x)) for x in X_feat_train])
    # y_pred = np.array([np.argmax(np.dot(np.transpose(w), x)) for x in AAT])
    # train_acc = [1 if y_pred[i] == y_train[i] else 0 for i in range(len(y_pred))]
    train_result = np.array([np.dot(np.transpose(w), x) for x in AAT])
    for t in threshold:
        inds = np.argwhere(train_result > t)
        y_pred = np.zeros(y_train.shape)
        for i in inds:
            y_pred[i[0], i[1]] = 1
        for r in range(len(y_pred)):
            if 1 not in y_pred[r]:
                y_pred[r][np.argmax(train_result[r])] = 1
        # train_acc = [1 if np.array_equal(y_pred[i], y_train[i]) else 0 for i in range(len(y_pred))]
        # train_acc = sum(train_acc)/len(y_pred)
        train_acc = average_precision_score(y_train, y_pred, average='micro')
        # train_acc2 = average_precision_score(y_train, y_pred, average='weighted')
        print("Threshold: " + str(t))
        print("Training Accuracy is " + str(train_acc))
        # print("Weighted: " + str(train_acc2))
    # y_pred = np.array([np.argmax(np.dot(np.transpose(w), x)) for x in X_feat_test])
    # y_pred = np.array([np.argmax(np.dot(np.transpose(w), x)) for x in test_XT])
    # acc =[1 if y_pred[i] == y_test[i] else 0 for i in range(len(y_pred))]
    test_result = np.array([np.dot(np.transpose(w), x) for x in test_XT])
    for t in threshold:
        inds = np.argwhere(test_result > t)
        y_pred = np.zeros(y_test.shape)
        for i in inds:
            y_pred[i[0], i[1]] = 1
        for r in range(len(y_pred)):
            if 1 not in y_pred[r]:
                y_pred[r][np.argmax(test_result[r])] = 1
        # for r in range(len(y_pred)):
        #     th = t * np.amax(test_result[r])
        #     for j in range(len(test_result[r])):
        #         if test_result[r][j] > th:
        #             y_pred[r][j] = 1

        # acc = [1 if y_test[i][np.argmax(test_result[i])] == 1 else 0 for i in range(len(y_pred))]
        # acc = [1 if np.array_equal(y_pred[i], y_test[i]) else 0 for i in range(len(y_pred))]
        # acc = sum(acc)/len(y_pred)
        acc = average_precision_score(y_test, y_pred, average='micro')
        # acc2 = average_precision_score(y_test, y_pred, average='weighted')
        print("Threshold: " + str(t))
        print("Test Accuracy is " + str(acc))
        # print("Weighted: " + str(acc2))
        save_indices.append(acc)
        save_labels.append(y_pred)
print(np.amax(np.array(save_indices)))
np.savetxt('patch6_4k_labels.csv', save_labels[np.argmax(np.array(save_indices))], delimiter=',')
