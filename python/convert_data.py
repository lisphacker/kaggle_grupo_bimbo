#!/usr/bin/env python

import pandas as pd
import numpy as np

def load_train_csv_file():
    return pd.read_csv('../data/original/train.csv', sep=',', header=0,
                       names=['week_num', 'sales_depot_id', 'sales_channel_id', 'route_id', 'client_id', 'product_id',
                              'sales_units_this_week', 'sales_this_week', 'returns_units_next_week', 'returns_next_week',
                              'adjusted_demand'],
                       dtype={'week_num':np.int32,
                              'sales_depot_id':np.int32,
                              'sales_channel_id':np.int32,
                              'route_id':np.int32,
                              'client_id':np.int32,
                              'product_id':np.int32,
                              'sales_units_this_week':np.int32,
                              'sales_this_week':np.float32,
                              'returns_units_next_week':np.int32,
                              'returns_next_week':np.float32,
                              'adjusted_demand':np.int32})
def load_test_csv_file():
    return pd.read_csv('../data/original/test.csv', sep=',', header=0,
                       names=['id', 'week_num', 'sales_depot_id', 'sales_channel_id',
                              'route_id', 'client_id', 'product_id'],
                       dtype={'id':np.int32,
                              'week_num':np.int32,
                              'sales_depot_id':np.int32,
                              'sales_channel_id':np.int32,
                              'route_id':np.int32,
                              'client_id':np.int32,
                              'product_id':np.int32})


def load_client_csv_file():
    return pd.read_csv('../data/original/cliente_tabla.csv', sep=',', header=0,
                       names=['client_id', 'client_name'],
                       dtype={'client_id':np.int32,
                              'client_name':np.str_})

def load_product_csv_file():
    return pd.read_csv('../data/original/producto_tabla.csv', sep=',', header=0,
                       names=['product_id', 'product_name'],
                       dtype={'product_id':np.int32,
                              'product_name':np.str_})

def load_town_state_csv_file():
    return pd.read_csv('../data/original/town_state.csv', sep=',', header=0,
                       names=['sales_depot_id', 'town', 'state'],
                       dtype={'sales_depot_id':np.int32,
                              'town':np.str_,
                              'state':np.str_})
                       
def clean_duplicates(train):
    return train.groupby(['client_id', 'product_id', 'week_num']).first()
