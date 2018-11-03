import numpy as np
import random
import struct

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from features import get_features, get_simple_features

def read_idx(filename):
    with open(filename, 'rb') as f:
        zero, data_type, dims = struct.unpack('>HBB', f.read(4))
        shape = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dims))
        return np.fromstring(f.read(), dtype=np.uint8).reshape(shape)


# Hyperparameters
reg = 1
patch_shape = (6,6)
n_features = 1024+512
block = 512
pool_size = 2

train_name = 'train-images-idx3-ubyte'
train_label_name = 'train-labels-idx1-ubyte'
test_name = 't10k-images-idx3-ubyte'
test_label_name = 't10k-labels-idx1-ubyte'

X_train = read_idx(train_name) / 255.
y_train = read_idx(train_label_name)
X_test = read_idx(test_name) / 255.
y_test = read_idx(test_label_name)

img_shape = X_train[0].shape

X_train = X_train.reshape((len(X_train), -1))
X_test = X_test.reshape((len(X_test), -1))

X_feat_train, X_feat_test = get_features(X_train, X_test, img_shape, n_features, block, patch_shape, pool_size)

# X_feat_train, X_feat_test = get_simple_features(X_train, X_test) #.9758 acc when 4096 features

# clf = SVC()
clf = LogisticRegression(C=reg)
print("Fitting")
clf.fit(X_feat_train, y_train)
print("Predicting")
y_pred = clf.predict(X_feat_test)
acc =[1 if y_pred[i] == y_test[i] else 0 for i in range(len(y_pred))]
acc = sum(acc)/len(y_pred)
print("Accuracy is " + str(acc)) # 0.9907 pytorch features
