import numpy as np, scipy as scp, random
import torch
import sys
import time
from sklearn.preprocessing import scale

from coatesng import BasicCoatesNgNet

# https://stackoverflow.com/questions/31528800/how-to-implement-zca-whitening-python
def ZCA(X):
    sigma = np.cov(X, rowvar=True)
    U, S, V = np.linalg.svd(sigma)
    # epsilon = 0.00001
    epsilon = 0.001
    ZCAMatrix = np.dot(U, np.dot(np.diag(1.0/np.sqrt(S + epsilon)), U.T))
    return ZCAMatrix

# https://stackoverflow.com/questions/16774148/fast-way-to-slice-image-into-overlapping-patches-and-merge-patches-to-image/16788733
def patchify(img, patch_shape, img_shape):
    img = np.ascontiguousarray(img)

    if patch_shape[2] == 1:
        X, Y = img_shape[0], img_shape[1]
        x, y = patch_shape[0], patch_shape[1]
        shape = (X - x + 1, Y - y + 1, x, y)
        strides = img.itemsize * np.array([Y, 1, Y, 1])
        patches = np.lib.stride_tricks.as_strided(img, shape=shape, strides=strides)
        return patches.reshape((patches.shape[0] * patches.shape[1], patch_shape[0], patch_shape[1]))
    elif len(patch_shape) == 3:
        X, Y, Z = img_shape[0], img_shape[1], img_shape[2]
        x, y, z = patch_shape[0], patch_shape[1], img_shape[2]
        strides = img.itemsize * np.array([Y*Z, Z, 1, Y*Z, Z, 1])
        shape = (X - x + 1, Y - y + 1, Z - z + 1, x, y, z)
        patches = np.lib.stride_tricks.as_strided(img, shape=shape, strides=strides)
        return patches.reshape(np.r_[-1, patch_shape])
    else:
        print("Channel mismatch")
        sys.exit()

def pytorch_features(X, patches, img_shape, patch_shape, block_size, pool_size, mode='train', vis_size=0):
    filters = patches.reshape(len(patches), patch_shape[0], patch_shape[1], patch_shape[2]).transpose(0,3,1,2)
    pool_kernel_size = int(np.ceil((img_shape[0] - patch_shape[0] + 1) / pool_size))
    pool_stride = pool_kernel_size
    # pool_kernel_size = 149 # 141
    # pool_stride = 51 # 55
    # pool_kernel_size = 75 #59 # 69
    # pool_stride = 25 # 32 # 27
    if mode=='visualize':
        pool_kernel_size = int(np.ceil((img_shape[0]-patch_shape[0] + 1) / vis_size))
        pool_stride = pool_kernel_size
    fbs = 8
    net = BasicCoatesNgNet(filters, patch_size=patch_shape[0], in_channels=patch_shape[2], pool_size=pool_kernel_size, pool_stride=pool_stride, bias=1.0, filter_batch_size=fbs)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    lift = None
    b_size = block_size
    blocks = int(np.ceil(len(X)/b_size))
    start = time.time()
    for i in range(blocks):
        print(i*10)
        if lift is None:
            lift = net(torch.from_numpy(X[i*b_size:(i+1)*b_size].reshape(-1,img_shape[0],img_shape[1],img_shape[2]).transpose(0,3,1,2)).to(device)).cpu().detach().numpy()
        else:
            lift = np.vstack((lift, net(torch.from_numpy(X[i*b_size:(i+1)*b_size].reshape(-1,img_shape[0],img_shape[1],img_shape[2]).transpose(0,3,1,2)).to(device)).cpu().detach().numpy()))
        # print(lift.shape)
    # lift = net(torch.from_numpy(X.reshape(len(X), img_shape[2], img_shape[0], img_shape[1])).to(device)).detach().numpy()
    # print(lift.shape)
    print(time.time()-start)
    return lift

def get_features(X_train, X_test, img_shape, n_features, block_size, patch_shape, pool_size):
    if len(patch_shape) == 2:
        patch_shape = np.r_[patch_shape, 1]
    if len(img_shape) == 2:
        img_shape = np.r_[img_shape, 1]
    print('Get Patches')
    X_train_c = X_train.copy()
    X_test_c = X_test.copy()
    patches_train = np.array([patchify(x, patch_shape, img_shape) for x in X_train_c])
    print(patches_train.shape)

    print("Whiten")
    # indices = np.random.choice(range(len(patches_train.reshape(-1, int(np.prod(patch_shape))))), 10000, replace=False)
    patches = patches_train.reshape(-1, int(np.prod(patch_shape)))
    # patches = scale(patches)
    whitener = ZCA(patches.T)
    print(whitener.shape)
    # patches_train = np.dot(patches_train.reshape(-1, int(np.prod(patch_shape))), whitener.T).reshape(patches_train.shape)
    patches_train = np.dot(np.dot(patches, whitener), whitener.T).reshape(patches_train.shape)
    print(patches_train.shape)

    indices = np.random.choice(range(len(patches_train.reshape(-1, int(np.prod(patch_shape))))), n_features, replace=False)
    patches_train = patches_train.reshape(-1, int(np.prod(patch_shape)))[indices]

    print('Convolve Patches to Features')
    b_size = 1024
    blocks = int(np.ceil(len(patches_train)/b_size))
    X_lift_train = None
    for i in range(blocks):
        print(i)
        if X_lift_train is None:
            X_lift_train = pytorch_features(X_train, patches_train[i*b_size:(i+1)*b_size], img_shape, patch_shape, block_size, pool_size)
            X_lift_test = pytorch_features(X_test, patches_train[i*b_size:(i+1)*b_size], img_shape, patch_shape, block_size, pool_size)
        else:
            X_lift_train = np.hstack((X_lift_train, pytorch_features(X_train, patches_train[i*b_size:(i+1)*b_size], img_shape, patch_shape, block_size, pool_size)))
            X_lift_test = np.hstack((X_lift_test, pytorch_features(X_test, patches_train[i*b_size:(i+1)*b_size], img_shape, patch_shape, block_size, pool_size)))
    # X_lift_train = pytorch_features(X_train, patches_train, img_shape, patch_shape, block_size, pool_size)
    print(X_lift_train.shape)
    # X_lift_test = pytorch_features(X_test, patches_train, img_shape, patch_shape, block_size, pool_size)

    return (X_lift_train.reshape(len(X_train), -1), X_lift_test.reshape(len(X_test), -1))

def get_features_repeat(X_train, X_test, img_shape, n_features, block_size, patch_shape, pool_size, patches_train, indices, mode='train', vis_size=0):
    patches_train = patches_train.reshape(-1, int(np.prod(patch_shape)))[indices]
    pool_size_test = pool_size
    print('Convolve Patches to Features')
    b_size = 1024
    blocks = int(np.ceil(len(patches_train)/b_size))
    X_lift_train = None
    for i in range(blocks):
        print(i)
        if X_lift_train is None:
            X_lift_train = pytorch_features(X_train, patches_train[i*b_size:(i+1)*b_size], img_shape, patch_shape, block_size, pool_size)
            X_lift_test = pytorch_features(X_test, patches_train[i*b_size:(i+1)*b_size], img_shape, patch_shape, block_size, pool_size, mode, vis_size)
        else:
            X_lift_train = np.hstack((X_lift_train, pytorch_features(X_train, patches_train[i*b_size:(i+1)*b_size], img_shape, patch_shape, block_size, pool_size)))
            X_lift_test = np.hstack((X_lift_test, pytorch_features(X_test, patches_train[i*b_size:(i+1)*b_size], img_shape, patch_shape, block_size, pool_size, mode, vis_size)))
    # X_lift_train = pytorch_features(X_train, patches_train, img_shape, patch_shape, block_size, pool_size)
    print(X_lift_train.shape)
    print(X_lift_test.shape)
    # X_lift_test = pytorch_features(X_test, patches_train, img_shape, patch_shape, block_size, pool_size)

    return (X_lift_train.reshape(len(X_train), -1), X_lift_test.reshape(len(X_test), -1))

def get_simple_features(X_train, X_test, feats):
    W = np.random.randn(len(X_train[0]), feats)
    X_train_lift = np.maximum(np.dot(X_train, W), 0)
    X_test_lift = np.maximum(np.dot(X_test, W), 0)
    print(X_train.shape)
    print(X_test.shape)
    print(X_train_lift.shape)
    print(X_test_lift.shape)
    return (X_train_lift, X_test_lift)
