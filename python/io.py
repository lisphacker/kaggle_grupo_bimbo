import pandas as pd

def load_training_data():
    return pd.read_hdf('../data/python/train.h5')

def load_testing_data():
    return pd.read_hdf('../data/python/test.h5')

def load_products():
    return pd.read_hdf('../data/python/products.h5')

def load_clients():
    return pd.read_hdf('../data/python/clients.h5')

def load_town_state():
    return pd.read_hdf('../data/python/town_state.h5')

def load():
    global train_full, test_full, products, clients, town_state

    train_full = load_training_data()
    test_full = load_testing_data()
    products = load_products()
    clients = load_clients()
    town_state = load_town_state()
