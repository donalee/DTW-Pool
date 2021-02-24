import os
import numpy as np
from sktime.utils.load_data import load_from_tsfile_to_dataframe, load_from_arff_to_dataframe

def load_ucr_dataset(dataset, split):
    datadir = './data/Univariate'
    if split in ['TRAIN', 'TEST']:
        filename = dataset + '_' + split
        filepath = os.path.join(datadir, dataset, filename)
        if os.path.isfile(filepath + '.ts'):
            data, labels = load_from_tsfile_to_dataframe(filepath + '.ts')
        elif os.path.isfile(filepath + '.arff'):
            data, labels = load_from_arff_to_dataframe(filepath + '.arff')
        else:
            raise ValueError("Invalid dataset")
    else:
        raise ValueError("Invalid split value")

    return data, labels

def load_uea_dataset(dataset, split):
    datadir = './data/Multivariate'
    if split in ['TRAIN', 'TEST']:
        filename = dataset + '_' + split
        filepath = os.path.join(datadir, dataset, filename)
        if os.path.isfile(filepath + '.ts'):
            data, labels = load_from_tsfile_to_dataframe(filepath + '.ts')
        elif os.path.isfile(filepath + '.arff'):
            data, labels = load_from_arff_to_dataframe(filepath + '.arff')
        else:
            raise ValueError("Invalid dataset")
    else:
        raise ValueError("Invalid split value")

    return data, labels

