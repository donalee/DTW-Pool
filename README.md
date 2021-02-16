# Learnable Dynamic Temporal Pooling for Time Series Classification

This is the author code of ["Learnable Dynamic Temporal Pooling for Time Series Classification"](https://to-be-appeared).
We employ (and customize) the publicly availabe implementation of soft-dtw, please refer to https://github.com/Maghoumi/pytorch-softdtw-cuda.

## Overview

TO-BE-WRITTEN

## Running the codes

### STEP 1. Install the following python libraries / packages

- numpy
- numba
- sktime
- pytorch


### STEP 2. Download the benchmark datasets for time series classification

- We provide a small univariate time series dataset, `GunPoint`, as default.
- The datatsets can be downloaded from the UCR/UEA repository: http://www.timeseriesclassification.com.
- Place `DATASET_TRAIN.ts` and `DATASET_TEST.ts` files in `./data/Univariate/DATASET` or `./data/Multivariate/DATASET`.


### STEP 3. Train the CNN classifier with the DTP layer

```
python train_classifier.py
```
You can specify the details of the classifier and its optimization by input arguments
