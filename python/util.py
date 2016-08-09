import numpy as np
import pandas as pd

def compute_delta(train):
    train['delta'] = train.sales_units_this_week - train.adjusted_demand

def cat(series):
    return series.astype('category')

def sample(df, n):
    return df.iloc[np.random.randint(df.shape[0], size=n)]

def partition(df, partition_sizes):
    train_size, xver_size, test_size = partition_sizes
    total_size = train_size + xver_size + test_size

    num_rows = df.shape[0]
    
    train_size = num_rows * train_size / total_size
    xver_size = num_rows * xver_size / total_size
    test_size = num_rows * test_size / total_size

    indices = list(range(num_rows))
    
    train_indices = np.random.choice(indices, int(train_size))
    xver_indices = np.random.choice(indices, int(xver_size))
    test_indices = np.random.choice(indices, int(test_size))

    train = data.iloc[train_indices]
    xver = data.iloc[xver_indices]
    test = data.iloc[test_indices]
    
    return train, xver, test

