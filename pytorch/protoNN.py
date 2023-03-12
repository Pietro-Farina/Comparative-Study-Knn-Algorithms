# Copyright (c) 2023 Pietro Farina
# Licensed under the MIT license.
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.
import torch
import torch.nn as nn
import numpy as np
import sys
from sklearn.cluster import KMeans

class ProtoNN(nn.Module):
    def __init__(self, inputDimension, projectionDimension, numPrototypes,
                 numOutputLabels, gamma, W=None, B=None, Z=None, W_train=True, B_train=True, Z_train=True):
        '''
        Forward computation graph for ProtoNN.

        inputDimension: Input data dimension or feature dimension.
        projectionDimension: hyperparameter
        numPrototypes: hyperparameter
        numOutputLabels: The number of output labels or classes
        W, B, Z: Numpy matrices that can be used to initialize
            projection matrix(W), prototype matrix (B) and prototype labels
            matrix (B).
            Expected Dimensions:
                W   inputDimension (d) x projectionDimension (d_cap)
                B   projectionDimension (d_cap) x numPrototypes (m)
                Z   numOutputLabels (L) x numPrototypes (m)
        W_train, B_train, Z_train: booleans indicating if we want to train the parameter
            default True, used to see the difference in accuracy went not all params are trained
        '''
        super(ProtoNN, self).__init__()
        self.__d = inputDimension
        self.__d_cap = projectionDimension
        self.__m = numPrototypes
        self.__L = numOutputLabels

        self.W, self.B, self.Z = None, None, None
        self.W_train, self.B_train, self.Z_train = W_train, B_train, Z_train
        self.gamma = gamma

        self.__validInit = False
        self.__initWBZ(W, B, Z)
        self.__validateInit()

    def __validateInit(self):
        self.__validInit = False
        errmsg = "Dimensions mismatch! Should be W[d, d_cap]"
        errmsg+= ", B[d_cap, m] and Z[L, m]"
        d, d_cap, m, L, _ = self.getHyperParams()
        assert self.W.shape[0] == d, errmsg
        assert self.W.shape[1] == d_cap, errmsg
        assert self.B.shape[0] == d_cap, errmsg
        assert self.B.shape[1] == m, errmsg
        assert self.Z.shape[0] == L, errmsg
        assert self.Z.shape[1] == m, errmsg
        self.__validInit = True

    def __initWBZ(self, inW, inB, inZ):
        if inW is None:
            self.W = torch.randn([self.__d, self.__d_cap])
            if self.W_train:
                self.W = nn.Parameter(self.W)
        else:
            self.W = torch.from_numpy(inW.astype(np.float32))
            if self.W_train:
                self.W = nn.Parameter(self.W)

        if inB is None:
            self.B = torch.randn([self.__d_cap, self.__m])
            if self.B_train:
                self.B = nn.Parameter(self.B)
        else:
            self.B = torch.from_numpy(inB.astype(np.float32))
            if self.B_train:
                self.B = nn.Parameter(self.B)

        if inZ is None:
            self.Z = torch.randn([self.__L, self.__m])
            if self.Z_train:
                self.Z = nn.Parameter(self.Z)
        else:
            self.Z = torch.from_numpy(inZ.astype(np.float32))
            if self.Z_train:
                self.Z = nn.Parameter(self.Z)

    def getHyperParams(self):
        '''
        Returns the model hyperparameters:
            [inputDimension, projectionDimension, numPrototypes,
            numOutputLabels, gamma]
        '''
        d =  self.__d
        dcap = self.__d_cap
        m = self.__m
        L = self.__L
        return d, dcap, m, L, self.gamma

    def getModelMatrices(self):
        '''
        Returns model matrices, which can then be evaluated to obtain
        corresponding numpy arrays.  These can then be exported as part of
        other implementations of ProtonNN, for instance a C++ implementation or
        pure python implementation.
        Returns
            [ProjectionMatrix (W), prototypeMatrix (B),
             prototypeLabelsMatrix (Z), gamma]
        '''
        return self.W, self.B, self.Z, self.gamma
    
    def save_model_to_file_c(self, file_name):
        '''
        Given the file name, save all the varaibles and parameters of
        the ProtoNN model needed in order to make predictions.
        Ideally the model can be saved in an .h file to be then
        imported by the main C program.
        '''
        values = self.getHyperParams()
        matrices = self.getModelMatrices()
        # remove the last value for matrices since is duplicated (GAMMA)
        matrices = matrices[:-1]
        try:
            with open(file_name, 'w') as f:
                print("#define INPUT_DIM %s\n" %(values[0]), file=f)
                print("#define PROJECTION_DIM %s\n" %(values[1]), file=f)
                print("#define NUM_PROTOTYPES %s\n" %(values[2]), file=f)
                print("#define NUM_OUTPUT %s\n" %(values[3]), file=f)
                print("#define GAMMA %s\n" %(values[4]), file=f)
                # W B Z
                for tensor in matrices:
                    print(tensor.shape)
                #W
                print("const float W_param[INPUT_DIM][PROJECTION_DIM] = {", file=f)
                flat = matrices[0].detach().numpy().ravel()
                with np.printoptions(floatmode='maxprec'):
                    print(*flat, sep=', ', file=f)
                print("};\n", file=f)
                
                # B
                print("const float B_param[PROJECTION_DIM][NUM_PROTOTYPES] = {", file=f)
                flat = matrices[1].detach().numpy().ravel()
                with np.printoptions(floatmode='maxprec'):
                    print(*flat, sep=', ', file=f)
                print("};\n", file=f)
                
                # Z
                print("const float Z_param[NUM_OUTPUT][NUM_PROTOTYPES] = {", file=f)
                flat = matrices[2].detach().numpy().ravel()
                with np.printoptions(floatmode='maxprec'):
                    print(*flat, sep=', ', file=f)
                print("};\n", file=f)
        except:
            msg = "An error occured while saving ProtoNN model to file"
            print(msg, file=sys.stderr)
    
    def getModelSizeAsDense(self):
        '''
        Returns the ProtoNN model size in KB considering the parameters 
        W, B and Z as dense matrix.
        Other values are saved as MACRO (not saved as variables in memory).
        This is not the space the model occupies during training, but the
        space needed to represent it (or to do prediction).
        We use KB = 1000 B, instead of KB = 1024 B, since is an upper limit
        independent of the architecture used.
        Returns
            size (KB)
        '''
        assert self.__validInit is True, "Initialization failed!"

        param_size = 0
        param_size += self.W.nelement() * self.W.element_size()
        param_size += self.B.nelement() * self.B.element_size()
        param_size += self.Z.nelement() * self.Z.element_size()
        size_kb = param_size / 1000
        return size_kb
    
    def initializePrototypesAndGamma(self, training_input, training_labels, input_W=None, sample=False, scaleZ=False, overrideGamma=True):
        '''
        Intialization of prototypes and Z
        Two options:
            - sample from training data set
            - k-means clustering on the transformed training data set. For Z:
                - pick the most common class in the cluster
                - sum all the score in the cluster and normalize
        Default: k-means for B and most common class in cluster for Z
        (current state: works only as default)
        In case of k-,eams, hyperparameter gamma is overwritten, if not indicated specifically,
        as 1/2.5*(median of distance between sample and closest cluster)
        trainin_input, training_labels: the training dataset
        input_W: numpy array of W, if not given will be used the one stored in the class.
        sample: option to initialize B as sampled from dataset
        scaleZ: option to initialize Z as rescale score of cluster.
        overrideGamma: option to set gamma heuristically
        '''
        d, d_cap, m, L, _ = self.getHyperParams()

        assert training_input.shape[1] == d, "Dimensions mismatch of input size!"
        assert training_labels.shape[1] == L, "Dimensions mismatch of labels size!"
        assert training_input.shape[0] == training_labels.shape[0],"Input and corresponding labels are in different numbers. Should be the same"
        
        if input_W is None:
            W_tensor = self.W.data.cpu().detach()
        else:
            errmsg = "Dimensions mismatch of given W! Should be [d, dcap]"
            assert input_W.shape[0] == d, errmsg
            assert input_W.shape[1] == d_cap, errmsg
            W_tensor = torch.from_numpy(input_W.astype(np.float32))
        
        # if (sample == False)
        transformed_train_input = torch.matmul(training_input.float(), W_tensor)
        # EVENTUALLY CHANGE RANDOM STATE WHEN NOT TESTING
        kmeans = KMeans(n_clusters=m, random_state=0, n_init=10, max_iter=50).fit(transformed_train_input)
        B_nparray = kmeans.cluster_centers_.T
        errmsg = "Dimensions mismatch of B! Should be [dcap, m]"
        assert B_nparray.shape[0] == d_cap, errmsg
        assert B_nparray.shape[1] == m, errmsg
        
        # if (scaleZ == False)
        score = torch.zeros([m, L], dtype=torch.float32)
        for j in range (0, training_labels.shape[0]):
            score[kmeans.labels_[j]] += training_labels[j]
        # change the most higher value for class of a prototype to one and zero the others
        z = 0
        for element in score:
            index = np.argmax(element)
            for value in element:
                z += value
            for x in range (0, L):
                if (x == index):
                    element[x] = 1
                else:
                    element[x] = 0
        assert z == kmeans.labels_.shape[0], "Sum of score of the prototypes should be equal to the number of sample"
        Z_tensor = score.T.detach()

        if self.W_train:
            self.W = nn.Parameter(self.W)
        else:
            self.W = W_tensor
        
        if self.B_train:
            self.B = nn.Parameter(torch.from_numpy(B_nparray.astype(np.float32)))
        else:
            self.B = torch.from_numpy(B_nparray.astype(np.float32))
        
        
        if self.Z_train:
            self.Z = nn.Parameter(self.Z)
        else:
            self.Z = Z_tensor


        # if (sample == False and overrideGamma
        if (overrideGamma):
            transformed_train_input = kmeans.transform(transformed_train_input)
            heursitic_gamma = np.median(transformed_train_input)
            heursitic_gamma = 1 / (2.5 * heursitic_gamma)
            self.gamma = heursitic_gamma

        self.__validateInit()

    def forward(self, X):
        '''
        This method is responsible for construction of the forward computation
        graph. The end point of the computation graph, or in other words the
        output operator for the forward computation is returned.

        X: Input of shape [-1, inputDimension]
        returns: The forward computation outputs, self.protoNNOut
        '''
        assert self.__validInit is True, "Initialization failed!"

        W, B, Z, gamma = self.W, self.B, self.Z, self.gamma
        WX = torch.matmul(X, W)
        dim = [-1, WX.shape[1], 1]
        WX = torch.reshape(WX, dim)
        dim = [1, B.shape[0], -1]
        B_ = torch.reshape(B, dim)
        l2sim = B_ - WX
        l2sim = torch.pow(l2sim, 2)
        l2sim = torch.sum(l2sim, dim=1, keepdim=True)
        self.l2sim = l2sim
        gammal2sim = (-1 * gamma * gamma) * l2sim
        M = torch.exp(gammal2sim)
        dim = [1] + list(Z.shape)
        Z_ = torch.reshape(Z, dim)
        y = Z_ * M
        y = torch.sum(y, dim=2)
        return y


