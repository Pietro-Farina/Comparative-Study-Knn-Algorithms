# Copyright (c) 2023 Pietro Farina
# Licensed under the MIT license.
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.
import torch
import numpy as np
import utils as utils


class ProtoNNTrainer:

    def __init__(self, protoNNObj, sparcityW, sparcityB, sparcityZ, regW=0, regB=0,
                 regZ=0, learningRate=1e-01, lossType='l2', device=None):
        '''
        A wrapper for the various techniques used for training ProtoNN. This
        subsumes both the responsibility of loss graph construction and
        performing training. The original training routine that is part of the
        C++ implementation of EdgeML used iterative hard thresholding (IHT),
        gamma estimation through median heuristic and other tricks for
        training ProtoNN. This module implements the same in pytorch
        and python.

        protoNNObj: An instance of ProtoNN class defining the forward
            computation graph. The loss functions and training routines will be
            attached to this instance.
        regW, regB, regZ: Regularization constants for W, B, and
            Z matrices of protoNN.
        sparcityW, sparcityB, sparcityZ: Sparsity constraints
            for W, B and Z matrices. A value between 0 (exclusive) and 1
            (inclusive) is expected. A value of 1 indicates dense training.
        learningRate: Initial learning rate for ADAM optimizer.
        X, Y : Placeholders for data and labels.
            X [-1, featureDimension]
            Y [-1, num Labels]
        lossType: ['l2', 'xentropy']

        '''
        self.protoNNObj = protoNNObj
        self.__regW = regW
        self.__regB = regB
        self.__regZ = regZ
        self.__sW = sparcityW
        self.__sB = sparcityB
        self.__sZ = sparcityZ
        self.__lR = learningRate
        self.sparseTraining = True
        if (sparcityW == 1.0) and (sparcityB == 1.0) and (sparcityZ == 1.0):
            self.sparseTraining = False
            #print("Sparse training disabled.", file=sys.stderr)
        self.W_th = None
        self.B_th = None
        self.Z_th = None
        self.__lossType = lossType
        self.optimizer = self.__optimizer()
        self.lossCriterion = None
        assert lossType in ['l2', 'xentropy']
        if lossType == 'l2':
            self.lossCriterion = torch.nn.MSELoss()
            #print("Using L2 (MSE) loss")
        else :
            self.lossCriterion = torch.nn.CrossEntropyLoss()
            #print("Using x-entropy loss")
        self.__validInit = False
        self.__validInit = self.__validateInit()
        if device is None:
            self.device = "cpu"
        else:
            self.device = device

    def __validateInit(self):
        assert self.__validInit == False
        msg = "Sparsity values should be between 0 and 1 (both inclusive)"
        assert 0 <= self.__sW <= 1, msg
        assert 0 <= self.__sB <= 1, msg
        assert 0 <= self.__sZ <= 1, msg
        return True

    def __optimizer(self):
        optimizer = torch.optim.Adam(self.protoNNObj.parameters(),
                                     lr=self.__lR)
        return optimizer

    def loss(self, logits, labels_or_target):
        labels = labels_or_target
        assert len(logits) == len(labels)
        assert len(labels.shape) == 2
        assert len(logits.shape) == 2
        regLoss = (self.__regW * (torch.norm(self.protoNNObj.W)**2) +
                   self.__regB * (torch.norm(self.protoNNObj.B)**2) +
                   self.__regZ * (torch.norm(self.protoNNObj.Z)**2))
        if self.__lossType == 'xentropy':
            _, labels = torch.max(labels, dim=1)
            assert len(labels.shape)== 1
        loss = self.lossCriterion(logits, labels) + regLoss
        return loss

    def accuracy(self, predictions, labels):
        '''
        Returns accuracy and number of correct predictions.
        '''
        assert len(predictions.shape) == 1
        assert len(labels.shape) == 1
        assert len(predictions) == len(labels)
        correct = (predictions == labels).double()
        numCorrect = torch.sum(correct)
        acc = torch.mean(correct)
        return acc, numCorrect
    
    def getModelSize(self):
        '''
        Returns the ProtoNN model size in KB taking into account the
        sparsity of the matrices W, B and Z.
        Other values are saved as MACRO (not saved as variables in memory).
        This is not the space the model occupies during training, but the
        space needed to represent it (or to do prediction).
        We use KB = 1000 B, instead of KB = 1024 B, since is an upper limit
        independent of the architecture used.
        Returns
            size (KB)
        '''
        assert self.__validInit is True, "Initialization ProtoNN Trainer failed!"

        W, B, Z, g = self.protoNNObj.getModelMatrices()
        param_size = 0
        param_size += min(1, 2*self.__sW) * W.nelement() * W.element_size()
        param_size += min(1, 2*self.__sB) * B.nelement() * B.element_size()
        param_size += min(1, 2*self.__sZ) * Z.nelement() * Z.element_size()
        size_kb = param_size / 1000
        return size_kb

    def hardThreshold(self):
        prtn = self.protoNNObj
        W, B, Z = prtn.W.data, prtn.B.data, prtn.Z.data
        newW = utils.hardThreshold(W, self.__sW, self.protoNNObj.W_train)
        newB = utils.hardThreshold(B, self.__sB, self.protoNNObj.B_train)
        newZ = utils.hardThreshold(Z, self.__sZ, self.protoNNObj.Z_train)
        prtn.W.data = torch.FloatTensor(newW).to(self.device)
        prtn.B.data = torch.FloatTensor(newB).to(self.device)
        prtn.Z.data = torch.FloatTensor(newZ).to(self.device)

    def train(self, x_train, x_val, y_train, y_val, batchSize=100, epochs=100, 
              printStep=1000, valStep=100, verbose=True, history=False):
        '''
        Performs dense training of ProtoNN followed by iterative hard
        thresholding to enforce sparsity constraints.

        batchSize: Batch size per update
        epochs : The number of epochs to run training for. One epoch is
            defined as one pass over the entire training data.
        x_train, x_val, y_train, y_val: The numpy array containing train and
            validation data. x data is assumed to in of shape [-1,
            featureDimension] while y should have shape [-1, numberLabels].
        printStep: Number of batches between echoing of loss and train accuracy.
        valStep: Number of epochs between evaluations on validation set.
        verbose: option to print additional info during training
        history: option to store the accuracy throgh epochs

        Returns a dictionary:
            'correctPredictions': number of correct predictions on test set
            'totalPredictions': number of prediction on test set
            'accuracyThroughEpochs': accuracy through epochs if history was set True,
                                     None otherwise
            'numberOfUpdate': the number of optimizing updates done on the parameters
        '''
        d, dcap, m, L, _ = self.protoNNObj.getHyperParams()
        assert batchSize >= 1, 'Batch size should be positive integer'
        assert epochs >= 1, 'Total epochs should be positive integer'
        assert x_train.ndim == 2, 'Expected training data to be of rank 2'
        assert x_train.shape[1] == d, 'Expected x_train to be [-1, %d]' % d
        assert x_val.ndim == 2, 'Expected validation data to be of rank 2'
        assert x_val.shape[1] == d, 'Expected x_val to be [-1, %d]' % d
        assert y_train.ndim == 2, 'Expected training labels to be of rank 2'
        assert y_train.shape[1] == L, 'Expected y_train to be [-1, %d]' % L
        assert y_val.ndim == 2, 'Expected validation labels to be of rank 2'
        assert y_val.shape[1] == L, 'Expected y_val to be [-1, %d]' % L

        trainNumBatches = int(np.ceil(len(x_train) / batchSize))
        valNumBatches = int(np.ceil(len(x_val) / batchSize))
        x_train_batches = np.array_split(x_train, trainNumBatches)
        y_train_batches = np.array_split(y_train, trainNumBatches)
        x_val_batches = np.array_split(x_val, valNumBatches)
        y_val_batches = np.array_split(y_val, valNumBatches)

        accuracyHistory = []
        lastCorrect = 0
        numberOfUpdates = len(x_train_batches)
        for epoch in range(epochs):
            for i in range(len(x_train_batches)):
                x_batch, y_batch = x_train_batches[i], y_train_batches[i]
                x_batch, y_batch = torch.Tensor(x_batch), torch.Tensor(y_batch)
                x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)
                self.optimizer.zero_grad()
                logits = self.protoNNObj.forward(x_batch)
                loss = self.loss(logits, y_batch)
                loss.backward()
                self.optimizer.step()
                _, predictions = torch.max(logits, dim=1)
                _, target = torch.max(y_batch, dim=1)
                acc, _ = self.accuracy(predictions, target)
                if (verbose and i % printStep == 0):
                    print("Epoch %d batch %d loss %f acc %f" % (epoch, i, loss,
                                                               acc))
            # Perform IHT Here.
            if self.sparseTraining:
                self.hardThreshold()
            # Perform Track of Data for each epoch
            if (history):
                numCorrect = 0
                for i in range(len(x_val_batches)):
                    x_batch, y_batch = x_val_batches[i], y_val_batches[i]
                    x_batch, y_batch = torch.Tensor(x_batch), torch.Tensor(y_batch)
                    x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)
                    logits = self.protoNNObj.forward(x_batch)
                    _, predictions = torch.max(logits, dim=1)
                    _, target = torch.max(y_batch, dim=1)
                    _, count = self.accuracy(predictions, target)
                    numCorrect += count
                accuracyHistory = accuracyHistory + [numCorrect / len(x_val)]
                lastCorrect = numCorrect
            # Perform validation set evaluation
            elif (epoch + 1) % valStep == 0:
                numCorrect = 0
                for i in range(len(x_val_batches)):
                    x_batch, y_batch = x_val_batches[i], y_val_batches[i]
                    x_batch, y_batch = torch.Tensor(x_batch), torch.Tensor(y_batch)
                    x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)
                    logits = self.protoNNObj.forward(x_batch)
                    _, predictions = torch.max(logits, dim=1)
                    _, target = torch.max(y_batch, dim=1)
                    _, count = self.accuracy(predictions, target)
                    numCorrect += count
                if (verbose):
                    print("Validation accuracy: %f" % (numCorrect / len(x_val)))
                lastCorrect = numCorrect
        result = {
            'correctPredictions': lastCorrect,
            'totalPredictions': len(x_val),
            'accuracyThroughEpochs': accuracyHistory,
            'numberOfUpdate': numberOfUpdates,
        }
        return result

