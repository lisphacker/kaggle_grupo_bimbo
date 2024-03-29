import sys
import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVR
from scipy.sparse import csr_matrix, lil_matrix

re_kg = re.compile(r'\b(\d+)[kK][gG]\b')
re_g = re.compile(r'\b(\d+)[gG]\b')

re_ml = re.compile(r'\b(\d+)ml\b')
re_l = re.compile(r'\b(\d+)[lL]\b')

re_vanilla = re.compile(r'(va.{0,1}nilla)')
re_chocolate = re.compile(r'(choc)')

def extract_weight(name):
    try:
        return float(re_kg.search(name).group(1))
    except:
        pass
    
    try:
        return float(re_g.search(name).group(1)) * 0.001
    except:
        pass

    return 0
    
def extract_volume(name):
    try:
        return float(re_ml.search(name).group(1)) * 0.001
    except:
        pass
    
    try:
        return float(re_l.search(name).group(1))
    except:
        pass

    return 0

def has_vanilla(lname):
    return re_vanilla.search(lname) is not None

def has_chocolate(lname):
    return re_chocolate.search(lname) is not None
    
def process_product_name(name):
    lname = name.lower()
    tokens = re.split('\W+', name)
    print(tokens, tokens[-2])

token_prefixes = ['ajon', 'amarant', 'animalito', 'arandano', 'azuc',
                  'barr', 'burrito',
                  'canel', 'caser', 'cereal', 'clasi', 'classic',
                  'danes',
                  'gran',
                  'integral', 'intens',
                  'median',
                  'natura', 'naviden',
                  'panet', 'prom',
                  'rebanad', 'redond',
                  'tortill', 'tostad',
                  'valent']

def clean_token(token):
    if token == 'empanzador':
        token = 'empanizador'

    if token == 'vainilla':
        token = 'vanilla'
        
    if token.endswith('s'):
        token = token[0:-1]
        
    if token.startswith('chispa'):
        token = 'chip'

    for prefix in token_prefixes:
        if token.startswith(prefix):
            token = prefix

    return token

    
def tokenize_product_name(name):
    re_num = re.compile(r'(\d+)')
    
    tokens = re.split('\W+', name.lower())
    tokens2 = []
    for token in tokens:
        if len(token) <= 3:
            continue

        if re_num.search(token) is not None:
            continue

        token = clean_token(token)
        
        tokens2.append(token)

    return tokens2
            
    
def get_word_freq(product_names):
    word_freq = {}

    re_num = re.compile(r'(\d+)')
    
    for pn in product_names:
        tokens = tokenize_product_name(pn)
        
        for token in tokens:
            new_freq = word_freq.get(token, 0) + 1
            word_freq[token] = new_freq

    return word_freq

def load_products():
    return pd.read_hdf('../data/python/products.h5')

def get_product_name_feature_matrix(products):
    cleaned_product_names = map(lambda pn: ' '.join(tokenize_product_name(pn)), products.product_name)
    
    vectorizer = TfidfVectorizer(input='content', dtype=np.float32, norm='l2')
    product_matrix = vectorizer.fit_transform(cleaned_product_names)
    return vectorizer, product_matrix

def make_id2idx_map(products):
    id2idx_map = {}
    
    for idx, id in enumerate(products.product_id):
        id2idx_map[id] = idx

    return id2idx_map
    
    
def make_wordvec_dataset(df, ps):
    for row in df.itertuples():
        print(row)
        print(row.product_id)

def build_trinout(vectorizer, product_matrix, id2idx_map, df, skip_output=False):
    num_features = len(vectorizer.get_feature_names())
    num_rows = df.shape[0]

    print('Creating SVM matrices')
    trin = lil_matrix((num_rows, num_features + 2), dtype='float32')
    if not skip_output:
        trout = np.ndarray(num_rows, dtype='float32')
    else:
        trout = None

    print('Populating SVM matrices')
    
    for i in range(num_rows):
        row = df.iloc[i]
        product_id = row.product_id
        trin[i, 0] = row.sales_units_this_week
        trin[i, 1] = row.returns_units_next_week
        trin[i, 2:(num_features+2)] = product_matrix[id2idx_map[product_id], :]

        if not skip_output:
            trout[i] = df.iloc[i].adjusted_demand

    return trin, trout
    
def train_svm(vectorizer, product_matrix, id2idx_map, df):
    trin, trout = build_trinout(vectorizer, product_matrix, id2idx_map, df)

    svm = SVR()
    #svm = SVR(kernel='sigmoid')

    print('Training SVM')
    svm.fit(trin, trout)

    return svm

def test_svm(svm, vectorizer, product_matrix, id2idx_map, df):
    trin, trout = build_trinout(vectorizer, product_matrix, id2idx_map, df)

    print('Predicting')
    trtest = svm.predict(trin)

    s = 0
    for test, out in zip(trtest, trout):
        if test < 0:
            test = 0
        d = np.log(test + 1) - np.log(out + 1)
        if np.isnan(d):
            print(test, out)

        s += d * d
        
    s /= len(trtest)
    print('Error =', np.sqrt(s))

def quick_test(vectorizer, product_matrix, id2idx_map, train, test):
    svm = train_svm(vectorizer, product_matrix, id2idx_map, train)
    test_svm(svm, vectorizer, product_matrix, id2idx_map, test)

def predict_svm(svm, vectorizer, product_matrix, id2idx_map, df):
    trin, _ = build_trinout(vectorizer, product_matrix, id2idx_map, df, True)

    print('Predicting')
    trout = svm.predict(trin)

    for id_, pred in zip(df.id, trout):
        print(id_, pred)


import code

class SVMModel:
    def __init__(self, products):
        self.svm = None

        self.vectorizer, self.product_matrix = get_product_name_feature_matrix(products)
        self.id2idx_map = make_id2idx_map(products)

    def train(self, train):
        self.build_training_data(train)
        
        input, output = self.build_svm_data(self.vectorizer, self.product_matrix, self.id2idx_map, train)
        
        self.svm = SVR()
        #svm = SVR(kernel='sigmoid')

        print('Training SVM')
        self.svm.fit(input, output)

    def test(self, test):
        input, ref = self.build_svm_data(self.vectorizer, self.product_matrix, self.id2idx_map, test)

        print('Predicting')
        output = self.svm.predict(input)

        #code.interact(local=locals())

        s = 0
        for r, o in zip(ref, output):
            if o < 0:
                print(r, o)
                o = 0
            d = np.log(o + 1) - np.log(r + 1)

            s += d * d
        
        s /= len(output)
        return np.sqrt(s)

    def build_training_data(self, train):
        print('Building training data')
        
        self.sch_prod_2_demand_map = dict()
        
        for row in train.itertuples():
            t = (row.client_id, row.product_id)
            if t not in self.sch_prod_2_demand_map:
                self.sch_prod_2_demand_map[t] = (row.adjusted_demand, 1)
            else:
                t2 = self.sch_prod_2_demand_map[t]
                self.sch_prod_2_demand_map[t] = (t2[0] + row.adjusted_demand, t2[1] + 1)

        for k in self.sch_prod_2_demand_map:
            t = self.sch_prod_2_demand_map[k]
            self.sch_prod_2_demand_map[k] = t[0] / t[1]
        
        
    def build_svm_data(self, vectorizer, product_matrix, id2idx_map, df, skip_output=False):
        num_features = len(vectorizer.get_feature_names())
        num_rows = df.shape[0]

        print('Creating SVM matrices')
        input = lil_matrix((num_rows, num_features + 2), dtype='float32')
        if not skip_output:
            output = np.ndarray(num_rows, dtype='float32')
        else:
            output = None

        print('Populating SVM matrices')
            
        for i in range(num_rows):
            row = df.iloc[i]
            product_id = row.product_id

            t = (row.client_id, row.product_id)
            if t in self.sch_prod_2_demand_map:
                demand = self.sch_prod_2_demand_map[t]
            else:
                demand = 0
                print(i, 0)
            
            input[i, 0] = demand
            input[i, 1] = row.client_id
            n = 2
            input[i, n:(num_features+n)] = product_matrix[id2idx_map[product_id], :]

            if not skip_output:
                output[i] = df.iloc[i].adjusted_demand

        return input, output

def qt():
    global products, train, test
    m = SVMModel(products)
    m.train(train)
    print(m.test(test))

    
