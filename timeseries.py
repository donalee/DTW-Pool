import os
import numpy as np
import torch
import torch.nn.functional as f
from torch.utils.data import Dataset
from utils import load_ucr_dataset, load_uea_dataset

class TimeSeriesWithLabels(Dataset):
    def __init__(self, dataset, datatype, split, **kwargs): 
        super().__init__()
        self.dataset = dataset

        if datatype == 'univar':
            data, labels = load_ucr_dataset(dataset, split)
        elif datatype == 'multivar':
            data, labels = load_uea_dataset(dataset, split)
        else:
            raise ValueError("Invalid vartype")
        self.data, self.labels = self._preprocess(data, labels)
        
        self.input_size = self.data.shape[1]
        self.num_classes = len(np.unique(labels))

    def _preprocess(self, data, labels):
        data = np.array([np.array([data.values[iidx, vidx].to_numpy(dtype=np.float) \
                                for vidx in range(data.values.shape[1])]) \
                                for iidx in range(data.values.shape[0])]) 
        data = torch.Tensor(data)
        data[torch.isnan(data)] = 0.0
        
        label2idx = {label: idx for idx, label in enumerate(np.unique(labels))}
        labels = np.array([label2idx[label] for label in labels])
        labels = torch.LongTensor(labels.astype(np.float))
        
        return data, labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {
            'data': self.data[idx],
            'labels': self.labels[idx],
        } 
