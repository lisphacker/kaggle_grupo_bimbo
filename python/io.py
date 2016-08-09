import pandas as pd

def load_training_data():
    return pd.read_hdf('../data/python/train.h5')

def load_testing_data():
    return pd.read_hdf('../data/python/test.h5')

def load_products():
    return pd.read_hdf('../data/python/products.h5')

