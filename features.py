import numpy as np, scipy as scp, random
import torch
from sklearn.kernel_approximation import RBFSampler
from scipy.signal import convolve2d
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
def patchify(img, patch_shape, img_shape):
    img = np.ascontiguousarray(img)
    X, Y = img_shape
    x, y = patch_shape
    shape = (X - x + 1, Y - y + 1, x, y)
    strides = img.itemsize * np.array([Y, 1, Y, 1])
    patches = np.lib.stride_tricks.as_strided(img, shape=shape, strides=strides)
    patches_shape = patches.shape
    return patches.reshape((patches_shape[0] * patches_shape[1], patch_shape[0], patch_shape[1]))

def pool(x, pool_size, im_shape):
    x = x.reshape(im_shape)
    if (im_shape[0] / 2 == int(im_shape[0] / 2)): #even
        x_pool = block_reduce(x, block_size = (int(np.ceil(im_shape[0]/pool_size)), int(np.ceil(im_shape[1]/pool_size))), func=np.max)
    else: #TODO split block correctly or pad?
        x_pool = block_reduce(x, block_size = (int(np.ceil(im_shape[0]/pool_size)), int(np.ceil(im_shape[1]/pool_size))), func=np.max)
    return np.ndarray.flatten(x_pool)

def pool_all(X, n, pool_size, im_shape):
    pooled = []
    for i in range(n):
        pooled.append(pool(X[:, i], pool_size, im_shape))
    return np.ndarray.flatten(np.array(pooled))

def pytorch_features(X, patches, patch_shape):
    sigma = 1.0
    filters = patches.reshape(len(patches), 1, patch_shape[0], patch_shape[0])
    net = BasicCoatesNgNet(filters, patch_size=patch_shape[0], in_channels=1, pool_size=12, pool_stride=12, bias=1.0, filter_batch_size=128)
    lift = None
    blocks = 40
    b_size = int(len(X)/blocks)
    for i in range(blocks):
        if lift is None:
            lift = net(torch.from_numpy(X[i*b_size:(i+1)*b_size].reshape(b_size,1,28,28))).detach().numpy()
        else:
            lift = np.vstack((lift, net(torch.from_numpy(X[i*b_size:(i+1)*b_size].reshape(b_size,1,28,28))).detach().numpy()))
        print(lift.shape)
    print(lift.shape)
    return lift

def get_features(X_train, X_test, img_shape, n_features, feature_block, patch_shape, pool_size):
    sampler = (RBFSampler(gamma=4, n_components=n_features)).fit(np.zeros((1, patch_shape[0] * patch_shape[1])))
    print(sampler.random_weights_.shape)
    print('Get Patches')
    X_train_c = X_train.copy()
    X_test_c = X_test.copy()
    patches_train = np.array([patchify(x, patch_shape, img_shape) for x in X_train_c])
    patches_test = np.array([patchify(x, patch_shape, img_shape) for x in X_test_c])
    print(patches_train.shape)

    print("Whiten")
    indices = np.random.choice(range(len(patches_train.reshape(-1, patch_shape[0]*patch_shape[1]))), 4096)
    patches = patches_train.reshape(-1, patch_shape[0]*patch_shape[1])[indices]
    whitener = ZCA(patches.T)
    print(whitener.shape)
    patches_train = np.dot(patches_train.reshape(-1, patch_shape[0]*patch_shape[1]), whitener.T).reshape(patches_train.shape)
    patches_test = np.dot(patches_test.reshape(-1, patch_shape[0]*patch_shape[1]), whitener.T).reshape(patches_test.shape)
    print(patches_train.shape)

    indices = np.random.choice(range(len(patches_train.reshape(-1, patch_shape[0]*patch_shape[1]))), n_features)
    patches_train = patches_train.reshape(-1, patch_shape[0]*patch_shape[1])[indices]

    print('Convolve Patches to Features')
    X_lift_train = pytorch_features(X_train, patches_train, patch_shape)
    print(X_lift_train.shape)
    X_lift_test = pytorch_features(X_test, patches_train, patch_shape)
    np.savetxt('lift_train.csv', X_lift_train.reshape(len(X_train), -1), delimiter=',')
    np.savetxt('lift_test.csv', X_lift_test.reshape(len(X_test), -1), delimiter=',')

    return (X_lift_train.reshape(len(X_train), -1), X_lift_test.reshape(len(X_test), -1))

def get_simple_features(X_train, X_test):
    W = np.random.randn(784, 2048)
    X_train_lift = np.maximum(X_train.dot(W), 0)
    X_test_lift = np.maximum(X_test.dot(W), 0)
    print(X_train.shape)
    print(X_test.shape)
    print(X_train_lift.shape)
    print(X_test_lift.shape)
    return (X_train_lift, X_test_lift)
