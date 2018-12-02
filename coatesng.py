import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import gc

def chunk_idxs_by_size(size, chunk_size):
    idxs = list(range(0, size+1, chunk_size))
    if (idxs[-1] != size):
        idxs.append(size)
    return list(zip(idxs[:-1], idxs[1:]))


class BasicCoatesNgNet(nn.Module):
    ''' All image inputs in torch must be C, H, W '''
    def __init__(self, filters, patch_size=6, in_channels=3, pool_size=2, pool_stride=2, bias=1.0, filter_batch_size=1024):
        super().__init__()
        self.pool_size = pool_size
        self.pool_stride = pool_stride
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.bias = bias
        self.filter_batch_size = filter_batch_size
        self.filters = filters.copy()
        self.active_filter_set = []
        self.start = None
        self.end  = None
        self.gpu = torch.cuda.is_available()

    def _forward(self, x):
        # Max pooling over a (2, 2) window
        if 'conv' not in self._modules:
            print(self.conv)
            raise Exception('No filters active, conv does not exist')
        conv = self.conv(x)
        x_pos = F.avg_pool2d(F.relu(conv - self.bias), [self.pool_size, self.pool_size],
                             stride=[self.pool_stride, self.pool_stride], ceil_mode=True)
        x_neg = F.avg_pool2d(F.relu((-1*conv) - self.bias) , [self.pool_size, self.pool_size],
                             stride=[self.pool_stride, self.pool_stride], ceil_mode=True)
        return torch.cat((x_pos, x_neg), dim=1)

    def forward(self, x):
        num_filters = self.filters.shape[0]
        activations = []
        for start, end in chunk_idxs_by_size(num_filters, self.filter_batch_size):
            activations.append(self.forward_partial(x, start, end))
        return torch.cat(activations , dim=1)
        self.conv = None

    def forward_partial(self, x, start, end):
        # We do this because gpus are horrible things
        gc.collect()
        self.activate(start,end)
        return self._forward(x)


    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

    def activate(self, start, end):
        if (self.start == start and self.end == end):
            return self
        self.start = start
        self.end = end
        filter_set = torch.from_numpy(self.filters[start:end])
        if (self.use_gpu):
            filter_set = filter_set.cuda()
        conv = nn.Conv2d(self.in_channels, end - start, self.patch_size, bias=False)
        conv.weight = nn.Parameter(filter_set)
        self.conv = None
        self.conv = conv
        self.active_filter_set = filter_set
        return self
    def deactivate(self):
        self.active_filter_set = None

if __name__ == "__main__":
    sigma = 1.0
    filters = sigma*np.random.randn(1024,3,6,6)
    net = BasicCoatesNgNet(filters)
    print(net)
