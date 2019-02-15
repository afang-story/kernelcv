import numpy as np
import scipy
import random
import sys
import csv

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
from voc_helpers.ptvoc import VOCClassification
# Parameters
experiment = 'VOC'
# reg = 1
threshold = [.2, .25, .33, .4, .5, .6, .66, .75, .8]
# threshold = [.3, .5, .7]
# n_features = 2*1024
n_features = 512 # 256
block_f = 128 # for visualize # 256 # for 256 x 256 and 6x6
block_n = 16
# block_f = 512 # for 128 x 128 and 6x6
# block_n = 32
# block_f = 1024
# block_n = 200
pool_size = 3
oversample = False
sgd_weights = False
visualize = True
save_visualize = True

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
    # transform = torchvision.transforms.CenterCrop(dim)
    transform = torchvision.transforms.Resize((dim, dim))
    # transform = torchvision.transforms.Compose([torchvision.transforms.Resize(dim), torchvision.transforms.CenterCrop(dim)])
    patch_shape = (6,6,3)
    # patch_shape2 = (12,12,3)
    yr = '2012'
    trainset = VOCClassification(root='./data', image_set='train', year=yr,
                                        download=True, transform=transform)
    # flip = torchvision.transforms.Compose([transform, torchvision.transforms.RandomHorizontalFlip(1)])
    # trainsetflip = VOCClassification(root='./data', image_set='train', year=yr, download=True, transform=flip)
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
    # X_train = np.vstack((X_train, trainsetflip.data))
    # y_train = np.vstack((y_train, np.array(trainsetflip.labels)))
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
levels = 0
# pyramids_train = [tuple(pyramid_gaussian(image, max_layer=levels, downscale=2, multichannel=True)) for image in X_train]
# pyramids_test = [tuple(pyramid_gaussian(image, max_layer=levels, downscale=2, multichannel=True)) for image in X_test]
train_sets = [X_train]
test_sets = [X_test]
# for i in range(1, levels+1):
#     train_sets.append(np.array([p[i] for p in pyramids_train]))
#     test_sets.append(np.array([p[i] for p in pyramids_test]))

img_shape = X_train[0].shape

X_train = X_train.reshape((len(X_train), -1))
X_test = X_test.reshape((len(X_test), -1))

if oversample:
    resampled = None
    resampled_labels = None
    class_freq = np.sum(y_train, axis=0)
    print(class_freq)
    for i in range(len(class_freq)):
        if class_freq[i] != np.amax(class_freq):
            diff = int(np.amax(class_freq) - class_freq[i])
            test_comp = np.zeros(class_freq.shape)
            test_comp[i] = 1
            class_samples = np.array([X_train[j] for j in range(len(X_train)) if np.array_equal(y_train[j], test_comp)])
            indices = np.random.choice(range(len(class_samples)), diff, replace=True)
            print(diff)
            if resampled is None:
                resampled = class_samples[indices]
                resampled_labels = np.tile(test_comp, (diff, 1))
            else:
                resampled = np.vstack((resampled, class_samples[indices]))
                resampled_labels = np.vstack((resampled_labels, np.tile(test_comp, (diff,1))))
    X_train = np.vstack((X_train, resampled))
    y_train = np.vstack((y_train, resampled_labels))
    y_train_ohe = y_train
    print(np.sum(y_train, axis=0))
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
train_lift = None
test_lift = None

if len(patch_shape) == 2:
    patch_shape = np.r_[patch_shape, 1]
if len(img_shape) == 2:
    img_shape = np.r_[img_shape, 1]

for d in range(levels+1):
    X_train = train_sets[d]
    X_test = test_sets[d]
    img_shape = X_train[0].shape
    X_train = X_train.reshape((len(X_train), -1))
    X_test = X_test.reshape((len(X_test), -1))
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
        if sgd_weights:
            if train_lift is None:
                train_lift = X_batch_train
                test_lift = X_batch_test
            else:
                train_lift = np.hstack((train_lift, X_batch_train))
                test_lift = np.hstack((test_lift, X_batch_test))
        AAT += np.dot(X_batch_train, X_batch_train.T)
        test_XT += np.dot(X_batch_test, X_batch_train.T)
save_indices = []
save_labels = []
save_prethresh = []
ws = []
thresh_used = []
print("Getting Matrix")
# regs = [1, 10, 100, 500, 1000, 10000, 100000, 1000000]
regs = [1, 10, 100, 500, 1000]
for reg in regs:
    print(reg)
    # w = scipy.linalg.solve(ATA + reg*np.identity(A.shape[1]), b, sym_pos=True)
    # w = np.dot(np.dot(A.T, np.linalg.inv(AAT + reg*np.identity(len(A)))), y_train_ohe)
    w = scipy.linalg.solve(AAT + reg*np.identity(AAT.shape[1]), y_train_ohe, sym_pos=True)
    if sgd_weights:
        true_w = np.dot(train_lift.T, w)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        dtype = torch.float64
        updated_dataset = torch.utils.data.TensorDataset(torch.tensor(train_lift, device=device, dtype=dtype), torch.tensor(y_train_ohe, device=device, dtype=dtype))
        updated_dataloader = torch.utils.data.DataLoader(updated_dataset, batch_size=128, shuffle=True, num_workers=0)
        net = nn.Sequential(nn.Linear(len(train_lift[0]), len(y_train_ohe[0]), bias=False))
        learning_rate = 0.001
        optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)
        def init_weights(m):
            m.weight = torch.nn.Parameter(torch.tensor(true_w.T, device=device, dtype=dtype, requires_grad=True))
        net.apply(init_weights)
        net.to(device)
        epochs = 15
        criterion = nn.BCEWithLogitsLoss()
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True, threshold=1e-4)
        for e in range(epochs):
            for t, (x, y) in enumerate(updated_dataloader):
                net.train()
                scores = net(x)
                loss = criterion(scores, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            if e % 5 == 0:
                print('Iteration %d, loss = %.4f' % (e, loss.item()))
                scheduler.step(loss)
        true_w = list(net.parameters())[0].data.cpu().numpy().T

    print("Predicting")
    # y_pred = np.array([np.argmax(np.dot(np.transpose(w), x)) for x in X_feat_train])
    # y_pred = np.array([np.argmax(np.dot(np.transpose(w), x)) for x in AAT])
    # train_acc = [1 if y_pred[i] == y_train[i] else 0 for i in range(len(y_pred))]
    if sgd_weights:
        train_result = np.array([np.dot(x, true_w) for x in train_lift])
    else:
        train_result = np.array([np.dot(np.transpose(w), x) for x in AAT])
    # train_result = softmax(train_result, axis=1)
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
    if sgd_weights:
        test_result = np.array([np.dot(x, true_w) for x in test_lift])
    else:
        test_result = np.array([np.dot(np.transpose(w), x) for x in test_XT])
    # test_result = softmax(test_result, axis=1)
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
        save_prethresh.append(test_result)
        ws.append(w)
        thresh_used.append(t)
print(np.amax(np.array(save_indices)))
AAT = None
test_XT = None
if visualize:
    w = ws[np.argmax(np.array(save_indices))]
    direct_w = None
    test_lift = None
    if save_visualize:
        np.savetxt('patch_visualize_labels.csv', save_labels[np.argmax(np.array(save_indices))], delimiter=',')
        np.savetxt('patch_visualize_prethresh.csv', save_prethresh[np.argmax(np.array(save_indices))], delimiter=',')
    for i in range(its):
        print(i)
        vis_size = 32
        X_batch_train, X_batch_test = get_features_repeat(X_train, X_test, img_shape, block_f, block_n, patch_shape, pool_size, patches_train, indices[i*block_f: (i+1)*block_f], mode='visualize', vis_size=vis_size)
        X_batch_train = np.float64(X_batch_train)
        X_batch_test = np.float64(X_batch_test)
        vis_size = int(np.sqrt(len(X_batch_test[0])/(2*block_f)))
        print(vis_size)
        print(len(X_batch_train))
        if direct_w is None:
            direct_w = np.dot(X_batch_train.T, w)
            test_lift = X_batch_test
        else:
            direct_w = np.hstack((direct_w, np.dot(X_batch_train.T, w)))
            test_lift = np.hstack((test_lift, X_batch_test))
    direct_w = direct_w.reshape((-1, pool_size, pool_size, len(w[0])))
    direct_w = np.transpose(direct_w, [3, 0, 1, 2])
    print(direct_w.shape)
    d1 = int(np.ceil(vis_size/pool_size))
    enlarge_w = np.zeros((len(direct_w), len(direct_w[0]), vis_size, vis_size))
    for a0 in range(len(direct_w)):
        for a1 in range(len(direct_w[0])):
            for a2 in range(vis_size):
                for a3 in range(vis_size):
                    enlarge_w[a0, a1, a2, a3] = direct_w[a0, a1, int(a2/d1), int(a3/d1)]
    print(enlarge_w.shape)
    '''
    enlarge_w = enlarge_w.transpose([1, 2, 3, 0])
    enlarge_w = enlarge_w.reshape((-1, len(w[0])))
    hist = []
    for t in range(len(test_lift)):
        # print(save_prethresh[np.argmax(np.array(save_indices))][t])
        # print(np.dot(test_lift[t], enlarge_w))
        asdf = save_prethresh[np.argmax(np.array(save_indices))][t]
        qwer = np.dot(test_lift[t], enlarge_w)
        hist += [qwer[z]/asdf[z] for z in range(len(asdf))]
    hist, bin_edges = np.histogram(hist, bins=[-1000, -800, -700, -600, -500, -400, -300, -200, -100, 0, 100, 200, 300, 400, 500, 600, 700, 800, 1000])
    print(hist)
    print(bin_edges)
    '''
    img_labels = []
    with open('data/files/VOC2012/classification_val.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count != 0:
                img_labels.append(row[0])
            line_count += 1
    test_lift = test_lift.reshape((len(test_lift), -1, vis_size, vis_size))
    print(test_lift.shape)
    for x in range(len(test_lift)):
        if x % 100 == 0:
            print(x)
        for cs in range(len(enlarge_w)):
            fname = 'visual/' + img_labels[x] + '_' + str(cs)
            temp = np.multiply(test_lift[x], enlarge_w[cs])
            temp = np.sum(temp, axis=0)
            # temp -= np.amin(temp)
            # temp /= np.amax(temp)
            if save_visualize:
                np.save(fname, temp)
        if save_visualize:
            fname2 = 'visualmax/' + img_labels[x]
            np.save(fname2, np.amax(test_lift[x], axis=0))
print("Achieved best val acc: " + str(np.amax(np.array(save_indices))))
print("Used threshold value: " + str(thresh_used[np.argmax(np.array(save_indices))]))
# np.savetxt('patch6_256_labels_pyramid_prethresh.csv', save_prethresh[np.argmax(np.array(save_indices))], delimiter=',')
# np.savetxt('patch6_256_labels_pyramid.csv', save_labels[np.argmax(np.array(save_indices))], delimiter=',')
