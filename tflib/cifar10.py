import numpy as np

import os
import urllib.request, urllib.parse, urllib.error
import gzip
import pickle as pickle

def unpickle(file):
    fo = open(file, 'rb')
    dict = pickle.load(fo)
    fo.close()
    return dict

def cifar_generator(filenames, batch_size, data_dir):
    all_data = []
    all_label = []
    for filename in filenames:
        dict = unpickle(data_dir + '/' + filename)
        all_data.append(dict['data'])
        all_label.append(dict['labels'])

    images = np.concatenate(all_data, axis=0)
    labels = np.concatenate(all_label,axis=0)
    print(images.shape)
    print(labels.shape)

    def get_epoch():
        rng_state = np.random.get_state()
        np.random.shuffle(images)
        np.random.set_state(rng_state)
        np.random.shuffle(labels)

        for i in range(len(images) / batch_size):
            yield np.copy(images[i*batch_size:(i+1)*batch_size]), np.copy(labels[i*batch_size:(i+1)*batch_size])

    return get_epoch

def splitGenerator(batch_size,data_dir):
    filenames = ['data_batch_1','data_batch_2','data_batch_3','data_batch_4','data_batch_5','test_batch']
    all_data = []
    all_label = []
    for filename in filenames:
        dict = unpickle(data_dir + '/' + filename)
        all_data.append(dict['data'])
        all_label.append(dict['labels'])

    images = np.concatenate(all_data, axis=0)
    labels = np.concatenate(all_label, axis=0)
    rng_state = np.random.get_state()
    np.random.shuffle(images)
    np.random.set_state(rng_state)
    np.random.shuffle(labels)
    data_size = len(labels)

    def getEpoch(num):
        length = data_size/(batch_size*10)
        i=num*length
        while True:
            yield np.copy(images[i * batch_size:(i + 1) * batch_size]), np.copy(labels[i * batch_size:(i + 1) * batch_size])
            i=i+1
            if i==(num+1)*length:
                i=num*length
    return [getEpoch(0),getEpoch(1),getEpoch(2),getEpoch(3),getEpoch(4),getEpoch(5),getEpoch(6),getEpoch(7),getEpoch(8),getEpoch(9)]



def load(batch_size, data_dir):
    return (
        cifar_generator(['data_batch_1','data_batch_2','data_batch_3','data_batch_4','data_batch_5'], batch_size, data_dir),
        cifar_generator(['test_batch'], batch_size, data_dir)
    )

def getTrainData(data_dir='/data/zhouwj/Dataset/cifar'):
    filenames = ['data_batch_1','data_batch_2','data_batch_3','data_batch_4','data_batch_5']
    all_data = []
    all_label = []
    for filename in filenames:
        dict = unpickle(data_dir + '/' + filename)
        all_data.append(dict['data'])
        all_label.append(dict['labels'])

    images = np.concatenate(all_data, axis=0)
    labels = np.concatenate(all_label,axis=0)

    return images,labels