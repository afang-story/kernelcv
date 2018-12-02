import numpy as np, scipy as scp, random
import torch
import sys

from skimage.measure import block_reduce
from coatesng import BasicCoatesNgNet

# https://stackoverflow.com/questions/31528800/how-to-implement-zca-whitening-python
def ZCA(X):
    sigma = np.cov(X, rowvar=True)
    U, S, V = np.linalg.svd(sigma)
    epsilon = 0.00001
    ZCAMatrix = np.dot(U, np.dot(np.diag(1.0/np.sqrt(S + epsilon)), U.T))
    return ZCAMatrix

# https://stackoverflow.com/questions/16774148/fast-way-to-slice-image-into-overlapping-patches-and-merge-patches-to-image/16788733
# https://stackoverflow.com/questions/42336919/how-to-extract-paches-from-3d-image-in-python
def patchify(img, patch_shape, img_shape):
    img = np.ascontiguousarray(img)

    if patch_shape[2] == 1:
        X, Y = img_shape[0], img_shape[1]
        x, y = patch_shape[0], patch_shape[1]
        shape = (X - x + 1, Y - y + 1, x, y)
        strides = img.itemsize * np.array([Y, 1, Y, 1])
        patches = np.lib.stride_tricks.as_strided(img, shape=shape, strides=strides)
        return patches.reshape((patches_shape[0] * patches_shape[1], patch_shape[0], patch_shape[1]))
    elif patch_shape[2] == 3:
        sh = np.array(img.shape)
        blck = np.asanyarray(patch_shape)
        strd = np.asanyarray([1, 1, 1])
        nbl = (sh - blck) // strd + 1
        strides = np.r_[img.strides * strd, img.strides]
        shape = np.r_[nbl, blck]
        patches = np.lib.stride_tricks.as_strided(img, shape=shape, strides=strides)
        return patches.reshape(np.r_[-1, patch_shape])
    else:
        print("Channel mismatch")
        sys.exit()


# def pool(x, pool_size, im_shape):
#     x = x.reshape(im_shape)
#     if (im_shape[0] / 2 == int(im_shape[0] / 2)): #even
#         x_pool = block_reduce(x, block_size = (int(np.ceil(im_shape[0]/pool_size)), int(np.ceil(im_shape[1]/pool_size))), func=np.max)
#     else: #TODO split block correctly or pad?
#         x_pool = block_reduce(x, block_size = (int(np.ceil(im_shape[0]/pool_size)), int(np.ceil(im_shape[1]/pool_size))), func=np.max)
#     return np.ndarray.flatten(x_pool)
#
# def pool_all(X, n, pool_size, im_shape):
#     pooled = []
#     for i in range(n):
#         pooled.append(pool(X[:, i], pool_size, im_shape))
#     return np.ndarray.flatten(np.array(pooled))

def pytorch_features(X, patches, img_shape, patch_shape, block_size, pool_size):
    sigma = 1.0
    filters = patches.reshape(len(patches), patch_shape[2], patch_shape[0], patch_shape[1])
    pool_kernel_size = int(np.ceil((img_shape[0] - patch_shape[0]) / pool_size))
    net = BasicCoatesNgNet(filters, patch_size=patch_shape[0], in_channels=patch_shape[2], pool_size=pool_kernel_size, pool_stride=pool_kernel_size, bias=1.0, filter_batch_size=128)
    lift = None
    blocks = 40
    b_size = int(len(X)/blocks)
    for i in range(blocks):
        if lift is None:
            lift = net(torch.from_numpy(X[i*b_size:(i+1)*b_size].reshape(b_size,img_shape[2],img_shape[0],img_shape[1]))).detach().numpy()
        else:
            lift = np.vstack((lift, net(torch.from_numpy(X[i*b_size:(i+1)*b_size].reshape(b_size,img_shape[2],img_shape[0],img_shape[1]))).detach().numpy()))
        print(lift.shape)
    print(lift.shape)
    return lift

def get_features(X_train, X_test, img_shape, n_features, block_size, patch_shape, pool_size):
    if len(patch_shape) == 2:
        patch_shape = np.r_[patch_shape, 1]
    if len(img_shape) == 2:
        image_shape = np.r_[img_shape, 1]
    print('Get Patches')
    X_train_c = X_train.copy()
    X_test_c = X_test.copy()
    patches_train = np.array([patchify(x, patch_shape, img_shape) for x in X_train_c])
    patches_test = np.array([patchify(x, patch_shape, img_shape) for x in X_test_c])
    print(patches_train.shape)

    print("Whiten")
    indices = np.random.choice(range(len(patches_train.reshape(-1, int(np.prod(patch_shape))))), 4096)
    patches = patches_train.reshape(-1, int(np.prod(patch_shape)))[indices]
    whitener = ZCA(patches.T)
    print(whitener.shape)
    patches_train = np.dot(patches_train.reshape(-1, int(np.prod(patch_shape))), whitener.T).reshape(patches_train.shape)
    patches_test = np.dot(patches_test.reshape(-1, int(np.prod(patch_shape))), whitener.T).reshape(patches_test.shape)
    print(patches_train.shape)

    indices = np.random.choice(range(len(patches_train.reshape(-1, int(np.prod(patch_shape))))), n_features)
    patches_train = patches_train.reshape(-1, int(np.prod(patch_shape)))[indices]

    print('Convolve Patches to Features')
    X_lift_train = pytorch_features(X_train, patches_train, img_shape, patch_shape, block_size, pool_size)
    print(X_lift_train.shape)
    X_lift_test = pytorch_features(X_test, patches_train, img_shape, patch_shape, block_size, pool_size)
    np.savetxt('lift_train.csv', X_lift_train.reshape(len(X_train), -1), delimiter=',')
    np.savetxt('lift_test.csv', X_lift_test.reshape(len(X_test), -1), delimiter=',')

    return (X_lift_train.reshape(len(X_train), -1), X_lift_test.reshape(len(X_test), -1))

def get_simple_features(X_train, X_test, feats):
    W = np.random.randn(784, feats)
    X_train_lift = np.maximum(np.dot(X_train, W), 0)
    X_test_lift = np.maximum(np.dot(X_test, W), 0)
    print(X_train.shape)
    print(X_test.shape)
    print(X_train_lift.shape)
    print(X_test_lift.shape)
    return (X_train_lift, X_test_lift)
