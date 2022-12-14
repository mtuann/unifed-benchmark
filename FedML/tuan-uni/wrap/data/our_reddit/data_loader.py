import json
import logging
import os

import numpy as np
import torch
import pandas as pd
import json

VOCAB_SIZE = 0
my_vocab = {}

def word_index_train(word):
    global VOCAB_SIZE
    global my_vocab

    if word in my_vocab:
        return my_vocab[word]
    else:
        my_vocab[word] = VOCAB_SIZE
        VOCAB_SIZE += 1
        return VOCAB_SIZE - 1

def word_index_test(word):
    global VOCAB_SIZE
    global my_vocab

    if word in my_vocab:
        return my_vocab[word]
    else:
        return 3

def trans_train(x):
    return np.array([[word_index_train(word) for word in x_item] for x_item in x], dtype = np.int64)

def trans_test(x):
    return np.array([[word_index_test(word) for word in x_item] for x_item in x], dtype = np.int64)


def read_data(train_data_dir, test_data_dir):
    global VOCAB_SIZE
    VOCAB_SIZE = 4
    global my_vocab
    my_vocab = {'<PAD>' : 0, '<BOS>' : 1, '<EOS>' : 2, '<OOV>' : 3}

    clients = []
    groups = []
    train_data,test_data = {},{}
    
    train_files = os.listdir(train_data_dir)
    train_files = [f for f in train_files if f.endswith('.json') and f[0]!='_']
    for f in train_files:
        client_name = f.split('.')[0]
        clients.append(client_name)
        train_data[client_name] = {'x':np.array([], dtype = np.int64).reshape(-1, 10), 'y':np.array([], dtype = np.int64).reshape(-1, 10)}

        file_path = os.path.join(train_data_dir, f)
        my_data = json.load(open(file_path, 'r'))["records"]

        for x in my_data:
            train_data[client_name]['y'] = np.vstack((train_data[client_name]['y'], trans_train(x[0]['target_tokens'])))
            train_data[client_name]['x'] = np.vstack((train_data[client_name]['x'], trans_train(x[1])))
            # print(client_name, train_data[client_name]['y'].shape, train_data[client_name]['x'].shape)

    
    test_files = os.listdir(test_data_dir)
    test_files = [f for f in test_files if f.endswith('.json') and f[0]!='_']
    for f in test_files:
        client_name = f.split('.')[0]
        test_data[client_name] = {'x':np.array([], dtype = np.int64).reshape(-1, 10), 'y':np.array([], dtype = np.int64).reshape(-1, 10)}

        file_path = os.path.join(test_data_dir, f)
        my_data = json.load(open(file_path, 'r'))["records"]
        
        for x in my_data:
            test_data[client_name]['y'] = np.vstack((test_data[client_name]['y'], trans_test(x[0]['target_tokens'])))
            test_data[client_name]['x'] = np.vstack((test_data[client_name]['x'], trans_test(x[1])))


    return clients, groups, train_data, test_data

def batch_data(data, batch_size):
    '''
    data is a dict := {'x': [numpy array], 'y': [numpy array]} (on one client)
    returns x, y, which are both numpy array of length: batch_size
    '''
    data_x = data['x']
    data_y = data['y']

    # randomly shuffle data
    np.random.seed(100)
    rng_state = np.random.get_state()
    np.random.shuffle(data_x)
    np.random.set_state(rng_state)
    np.random.shuffle(data_y)

    # loop through mini-batches
    batch_data = list()
    for i in range(0, len(data_x), batch_size):
        batched_x = data_x[i:i + batch_size]
        batched_y = data_y[i:i + batch_size]
        batched_x = torch.from_numpy(np.asarray(batched_x)).float()
        batched_y = torch.from_numpy(np.asarray(batched_y)).long()
        batch_data.append((batched_x, batched_y))
    return batch_data


def load_partition_data_reddit(batch_size,
                              train_path="../data/reddit/train/",
                              test_path="../data/reddit/test/"):
    users, groups, train_data, test_data = read_data(train_path, test_path)

    if len(groups) == 0:
        groups = [None for _ in users]
    train_data_num = 0
    test_data_num = 0
    train_data_local_dict = dict()
    test_data_local_dict = dict()
    train_data_local_num_dict = dict()
    train_data_global = list()
    test_data_global = list()
    client_idx = 0
    logging.info("loading data...")
    for u, g in zip(users, groups):
        user_train_data_num = len(train_data[u]['x'])
        user_test_data_num = len(test_data[u]['x'])
        train_data_num += user_train_data_num
        test_data_num += user_test_data_num
        train_data_local_num_dict[client_idx] = user_train_data_num

        # transform to batches
        train_batch = batch_data(train_data[u], batch_size)
        test_batch = batch_data(test_data[u], batch_size)

        # index using client index
        train_data_local_dict[client_idx] = train_batch
        test_data_local_dict[client_idx] = test_batch
        train_data_global += train_batch
        test_data_global += test_batch
        client_idx += 1
    logging.info("finished the loading data")
    client_num = client_idx
    global VOCAB_SIZE
    class_num = VOCAB_SIZE
    # to update the number of classes

    return client_num, train_data_num, test_data_num, train_data_global, test_data_global, \
           train_data_local_num_dict, train_data_local_dict, test_data_local_dict, class_num


# read_data('../../../../../flbenchmark/data/reddit/train', '../../../../../flbenchmark/data/reddit/test')
