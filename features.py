import numpy as np, scipy as scp, random
from sklearn.kernel_approximation import RBFSampler
from scipy.signal import convolve2d
from skimage.measure import block_reduce

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

def convolution(patches, weights, offset):
    print("in")
    patches = patches.reshape((patches.shape[0], patches.shape[1], -1))
    X_lift = np.zeros((len(patches), patches.shape[1], len(offset))) #60000 529 36
    print(patches.shape)
    for p in range(len(patches)):
        X_lift[p] = np.cos(np.dot(patches[p], weights) + offset)
    return np.array(X_lift)

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

    print('Convolve Patches to Features')
    b_size = int(len(patches_train)/5)
    X_lift_train = None
    for i in range(5):
        print(i)
        X_lift_train_b = None
        for j in range(int(n_features/feature_block)):
            weights = sampler.random_weights_[:, j*feature_block: (j+1)*feature_block]
            offset = sampler.random_offset_[j*feature_block: (j+1)*feature_block]
            X_lift_train_f = convolution(patches_train[i*b_size:(i+1)*b_size], weights, offset)
            if X_lift_train_b is None:
                X_lift_train_b = X_lift_train_f
            else:
                X_lift_train_b = np.hstack((X_lift_train_b, X_lift_train_f))
        if X_lift_train is None:
            X_lift_train = X_lift_train_b
        else:
            X_lift_train = np.vstack((X_lift_train, X_lift_train_b))

    # X_lift_test = convolution(patches_test, weights, offset)
    X_lift_test = None
    for j in range(int(n_features/feature_block)):
        weights = sampler.random_weights_[:, j*feature_block: (j+1)*feature_block]
        offset = sampler.random_offset_[j*feature_block: (j+1)*feature_block]
        X_lift_test_f = convolution(patches_test, weights, offset)
        if X_lift_test is None:
            X_lift_test = X_lift_test_f
        else:
            X_lift_test = np.hstack((X_lift_test, X_lift_test_f))
    print(X_lift_train.shape)

    print('Pool')
    im_shape = (int(np.sqrt(X_lift_train.shape[1])), int(np.sqrt(X_lift_train.shape[1])))
    X_lift_train = np.array([pool_all(x, len(sampler.random_offset_), pool_size, im_shape) for x in X_lift_train])
    X_lift_test = np.array([pool_all(x,len(sampler.random_offset_),pool_size, im_shape) for x in X_lift_test])
    print(X_lift_train.shape)
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
