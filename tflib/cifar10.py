import numpy as np

import os
import urllib
import gzip
import cPickle as pickle

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

    def get_epoch():
        rng_state = np.random.get_state()
        np.random.shuffle(images)
        np.random.set_state(rng_state)
        np.random.shuffle(labels)

        for i in xrange(len(images) / batch_size):
            yield np.copy(images[i*batch_size:(i+1)*batch_size]), np.copy(labels[i*batch_size:(i+1)*batch_size])

    return get_epoch


def load(batch_size, data_dir):
    return (
        cifar_generator(['data_batch_1','data_batch_2','data_batch_3','data_batch_4','data_batch_5'], batch_size, data_dir), 
        cifar_generator(['test_batch'], batch_size, data_dir)
    )
