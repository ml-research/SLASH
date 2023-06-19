import torch
from torch.utils.data import Dataset
import numpy as np

class MNIST_Addition(Dataset):

    def __init__(self, dataset, examples, num_i, flat_for_pc):
        self.data = list()
        self.dataset = dataset
        self.num_i = num_i
        self.flat_for_pc = flat_for_pc

        with open(examples) as f:
            for line in f:
                line = line.strip().split(' ')
                self.data.append(tuple([int(i) for i in line]))

    
    def __getitem__(self, index):
        if self.num_i == 2:
            i1, i2, l = self.data[index]
            l = ':- not addition(i1, i2, {}).'.format(l)
            if self.flat_for_pc:
                return {'i1': self.dataset[i1][0].flatten(), 'i2': self.dataset[i2][0].flatten()}, l
            else:
                return {'i1': self.dataset[i1][0], 'i2': self.dataset[i2][0]}, l

        elif self.num_i == 3:
            i1, i2, i3, l = self.data[index]
            l = ':- not addition(i1, i2, i3, {}).'.format(l)
            if self.flat_for_pc:
                return {'i1': self.dataset[i1][0].flatten(), 'i2': self.dataset[i2][0].flatten(), 'i3': self.dataset[i3][0].flatten()}, l
            else:
                return {'i1': self.dataset[i1][0], 'i2': self.dataset[i2][0], 'i3': self.dataset[i3][0]}, l

        elif self.num_i == 4:
            i1, i2, i3, i4, l = self.data[index]
            l = ':- not addition(i1, i2, i3, i4, {}).'.format(l)
            if self.flat_for_pc:
                return {'i1': self.dataset[i1][0].flatten(), 'i2': self.dataset[i2][0].flatten(), 'i3': self.dataset[i3][0].flatten(), 'i4': self.dataset[i4][0].flatten()}, l
            else:
                return {'i1': self.dataset[i1][0], 'i2': self.dataset[i2][0], 'i3': self.dataset[i3][0], 'i4': self.dataset[i4][0]}, l
    
        elif self.num_i == 6:
            i1, i2, i3, i4, i5,i6, l = self.data[index]
            l = ':- not addition(i1, i2, i3, i4, i5, i6, {}).'.format(l)
            if self.flat_for_pc:
                return {'i1': self.dataset[i1][0].flatten(), 'i2': self.dataset[i2][0].flatten(), 'i3': self.dataset[i3][0].flatten(), 'i4': self.dataset[i4][0].flatten(), 'i5': self.dataset[i5][0].flatten(), 'i6': self.dataset[i6][0].flatten()}, l
            else:
                return {'i1': self.dataset[i1][0], 'i2': self.dataset[i2][0], 'i3': self.dataset[i3][0], 'i4': self.dataset[i4][0], 'i5': self.dataset[i5][0], 'i6': self.dataset[i6][0]}, l
    

    def __len__(self):
        return len(self.data)