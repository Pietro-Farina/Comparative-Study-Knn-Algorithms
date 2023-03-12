# Copyright (c) 2023 Pietro Farina
# Licensed under the MIT license.
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler

def hardThreshold(A: torch.Tensor, s, isParam: bool=True):
    '''
    Hard thresholds and modifies the tensor A with sparsity s
    If the tensor A is a parameter, set the requires_grad as true for optimization
    '''
    A_ = A.data.cpu().detach().numpy().ravel()    
    if len(A_) > 0:
        th = np.percentile(np.abs(A_), (1 - s) * 100.0, interpolation='higher')
        A_[np.abs(A_) < th] = 0.0
    A_ = A_.reshape(A.shape)
    return torch.tensor(A_, requires_grad=isParam)

class CustomDatasetProtoNN(Dataset):
    '''
    sublcass of torch Dataset,
    It will transform the labels from a scalar into a space large as the number of
    different classes. For each label the corresponding index is set to 1, all others
    to 0.
    By default it will apply nomralization to the input dataset
    If the binary option is requested it will transform all classes after the first
    into a single one (first class Vs All)
    '''
    def __init__(self, data, labels, rescaling: bool=True, binary: bool=False):
        if (rescaling):
            data = StandardScaler().fit_transform(data)
            print("Dataset input rescaled")
        self.data = data
        if (binary):
            # binary
            self.numClasses = 2
        else:
            # multiclass
            self.numClasses = len(np.unique(labels))
        print("Dataset loaded, input space:", data.shape, ", output space:", labels.shape)
        # reshaping
        labels_reshaped = np.zeros((labels.shape[0], self.numClasses))
        for i in range (0, labels.shape[0]):
            if (binary):
                # binary (All class > 0 are saved as class 1)
                if (labels[i] == 0):
                    labels_reshaped[i][0] = 1
                else:
                    labels_reshaped[i][1] = 1
            else:
                # multiclass
                labels_reshaped[i][labels[i]] = 1
        print("Dataset reshaped, input space:", data.shape, ", output space:", labels_reshaped.shape)
        self.labels = labels_reshaped

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        out_data = self.data[idx]
        out_label = self.labels[idx]
        return out_data, out_label

def max_projection_dimensions(size, n_prototypes, input_dim, n_outputs, sw=1, sb=1, sz=1):
    '''
    Given:
        the maximum model size,
        the variables of the problem: input dimension and number of class,
        if specified the sparsity of the parameters,
        the chosen number of prototypes,
    Return maximum number of projection dimension that the ProtoNN can have while
    meeting the size constraints. If a negative value is returned it means that
    with this configuration the ProtoNN cannot respect the size restrictions.
    We consider 1 KB = 1000 B, instead of 1 KB = 1024 KB,
    since is an upper limit independent of the architecture used.
    '''
    assert size > 0, "Size should be greater then 0, expressed in KB"
    assert n_prototypes > 0, "Number of prototypes should be greater then 0"
    assert input_dim > 0, "Dimension of input should be greater then 0"
    assert n_outputs > 0, "Dimension of output should be greater then 0"
    result = ((size / 4) * 1000) - (min(1, 2*sz) * n_prototypes * n_outputs)
    div = ((min(1, 2*sw) *input_dim) + (min(1, 2*sb) *n_prototypes))
    result = result / div
    return result

def max_number_prototypes(size, projection_dim, input_dim, n_outputs, sw=1, sb=1, sz=1):
    '''
    Given:
        the maximum model size,
        the variables of the problem: input dimension and number of class,
        if specified the sparsity of the parameters,
        the chosen number of projection dimension.
    Return maximum number of prototypes that the ProtoNN can have while
    meeting the size constraints. If a negative value is returned it means that
    with this configuration the ProtoNN cannot respect the size restrictions.
    We consider 1 KB = 1000 B, instead of 1 KB = 1024 KB,
    since is an upper limit independent of the architecture used.
    '''
    assert size > 0, "Size should be greater then 0, expressed in KB"
    assert projection_dim > 0, "Projection dimension should be greater then 0"
    assert input_dim > 0, "Dimension of input should be greater then 0"
    assert n_outputs > 0, "Dimension of output should be greater then 0"    
    result = ((size / 4) * 1000) - (min(1, 2*sw) * projection_dim * input_dim)
    div = (((min(1, 2*sb) * projection_dim) + (min(1, 2*sz) *n_outputs)))
    result = result / div
    return result

# ---------- The following function are mainly used to summarize functions repeated in testing ---------- #
# ---------- They do not affect ProtoNN operations ---------- #
import pandas as pd
import h5py
import sys

def handleLetter26Dataset(filepath_or_buffer):
    '''
    Read the Letter26 dataset and return two numpy array X, Y containing
    data and the corresponding labels.
    Raise an exception if errors encountered
    '''
    try:
        data = pd.read_csv(filepath_or_buffer)
        data = data.replace({'label': {chr(i + 64): i-1 for i in range(1, 27)}})
        X = data.drop(columns='label')
        Y = data['label']
        X = np.array(X)
        Y = np.array(Y)
        return X, Y
    except:
        msg = "An error occured while reading the Letter26 dataset or while changing the labels from letters to integer\nConsider to manually import the dataset"
        print(msg, file=sys.stderr)
        return None, None
    
def handleMNIST(filepath_or_buffer):
    '''
    Read the MNIST dataset and return two numpy array X, Y containing
    data and the corresponding labels.
    Raise an exception if errors encountered
    '''
    try:
        data = pd.read_csv(filepath_or_buffer)
        X = data.drop(columns='label')
        Y = data['label']
        X = np.array(X)
        Y = np.array(Y)
        return X, Y
    except:
        msg = "An error occured while reading the MNIST dataset \nConsider to manually import the dataset"
        print(msg, file=sys.stderr)
        return None, None

def handleUSPS(filepath_or_buffer):
    '''
    Read the USPS dataset and return two numpy array X, Y containing
    data and the corresponding labels.
    Raise an exception if errors encountered
    '''
    try:
        with h5py.File(filepath_or_buffer, 'r') as hf:
            train = hf.get('train')
            X_tr = train.get('data')[:]
            Y_tr = train.get('target')[:]
            data_input = pd.DataFrame(data=X_tr)
            data_output = pd.DataFrame(data=Y_tr)
            X = np.array(data_input)
            Y = np.array(data_output)
            return X, Y
    except:
        msg = "An error occured while reading the USPS dataset \nConsider to manually import the dataset"
        print(msg, file=sys.stderr)
        return None, None

def knn_size(train_points, inputdim):
    '''
    Given the number of train points and their input dimension, return the size of the
    K-NN model in KB. We consider 1 KB = 1000 B, instead of 1 KB = 1024 KB,
    since is an upper limit independent of the architecture used
    '''
    nr_knn_points = train_points
    nr_elements = nr_knn_points * (inputdim + 1)
    nr_bytes = nr_elements * 4
    kb = nr_bytes / 1000
    print(kb)

