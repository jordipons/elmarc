import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import tensorflow as tf
import librosa
import pickle
import time
import os
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.cross_validation import PredefinedSplit
import datasets
import pandas as pd


config = {

    #'dataset': 'Extended Ballroom',
    #'audio_path': '/mnt/vmdata/users/jpons/extended_ballroom/',
    #'save_extracted_features_folder': '../data/Extended_Ballroom/features/', 
    #'results_folder': '../data/Extended_Ballroom/results/',
    #'train_set_list': None,
    #'val_set_list': None,
    #'test_set_list': None,
    #'audios_list': '/mnt/vmdata/users/jpons/extended_ballroom/all_files.txt', 
    # set 'audios_list' to FALSE to specify partitions in 'train/val/test_set_list'
    #'fix_length_by': 'zero-pad', # 'zero-pad', 'repeat-pad' or 'crop'

    #'dataset': 'Ballroom',
    #'audio_path': '/homedtic/jpons/ballroom/BallroomData/',
    #'save_extracted_features_folder': '../data/Ballroom/features/', 
    #'results_folder': '../data/Ballroom/results/',
    #'train_set_list': None,
    #'val_set_list': None,
    #'test_set_list': None,
    #'audios_list': '/homedtic/jpons/ballroom/allBallroomFiles.txt', 
    # set 'audios_list' to FALSE to specify partitions in 'train/val/test_set_list'
    #'fix_length_by': 'zero-pad', # 'zero-pad', 'repeat-pad' or 'crop'

    #'dataset': 'GTZAN',
    #'audio_path': '/data/GTZAN/',
    #'save_extracted_features_folder': '../data/GTZAN/features/', 
    #'results_folder': '../data/GTZAN/results/',
    #'train_set_list': '/home/jpons/GTZAN_partitions/train_filtered.txt',
    #'val_set_list': '/home/jpons/GTZAN_partitions/valid_filtered.txt',
    #'test_set_list': '/home/jpons/GTZAN_partitions/test_filtered.txt',
    #'audios_list': False, 
    # set 'audios_list' to FALSE to specify partitions in 'train/val/test_set_list'
    #'fix_length_by': 'zero-pad', # 'zero-pad', 'repeat-pad' or 'crop'

    'dataset': 'UrbanSound8K',
    'audio_path': '/data/UrbanSound8K/',
    'save_extracted_features_folder': '../data/UrbanSound8K/features/', 
    'results_folder': '../data/UrbanSound8K/results/',
    'train_set_list': None,
    'val_set_list': None,
    'test_set_list': None,
    'audios_list': '/data/UrbanSound8K/allFiles.txt', 
    # set 'audios_list' to FALSE to specify partitions in 'train/val/test_set_list'
    'fix_length_by': 'repeat-pad', # 'zero-pad', 'repeat-pad', 'crop' or False

    'debug': False,
    'sampling_rate': 12000,
    'experiment_name': 'v0',
    'features_type': 'MFCC', # CNN or MFCC
    'load_extracted_features': False,

    'CNN': {
        'n_mels': 96,
        'n_frames': 188, # GTZAN: 1404, old: 1360  ## BALLROOM: 1376  ### US8K: 100
        'batch_size': 5,

        #'architecture': 'cnn_small_filters',
        #'num_filters': 32, # 717, 90 or 32
        #'selected_features_list': [0, 1, 2, 3, 4]

        'architecture': 'cnn_music',
        'num_filters': 4, # 256, 128, 64, 32, 16, 8 or 4
        'selected_features_list': [0,1] # timbral [0], temporal [1] or both [0, 1]
    },
    'MFCC': {
        'number': 20,
        'fixed_length': 2048
    }
}

if config['debug'] and config['dataset']=='GTZAN':
    config['audio_path'] = '/home/jpons/GTZAN_debug/'
    if config['audios_list'] == False:
        config['train_set_list'] = '/home/jpons/GTZAN_debug_partitions/train_filtered.txt'
        config['val_set_list'] = '/home/jpons/GTZAN_debug_partitions/valid_filtered.txt'
        config['test_set_list'] = '/home/jpons/GTZAN_debug_partitions/test_filtered.txt'
    else:
        config['audios_list'] = '/home/jpons/GTZAN_debug_partitions/list.txt'

svm_params = [

    {'kernel': ['rbf'],
     'gamma': [1 / (2 ** 3), 1 / (2 ** 5), 1 / (2 ** 7), 1 / (2 ** 9), 1 / (2 ** 11), 1 / (2 ** 13), 'auto'],
     'C': [0.1, 2.0, 8.0, 32.0]},

    {'kernel': ['linear'],
     'C': [0.1, 2.0, 8.0, 32.0]}

]


# CNNs

def count_params(trainable_variables):
    "Return number of trainable variables, specifically: tf.trainable_variables()"
    return np.sum([np.prod(v.get_shape().as_list()) for v in trainable_variables])

def iterate_minibatches(prefix, audio_paths_list, batchsize):

    total_size = len(audio_paths_list)
    n_leftover = int(total_size % batchsize)
    leftover = n_leftover != 0

    for start_i in range(0, len(audio_paths_list) - batchsize + 1, batchsize):          
        first = True
        ground_truth = []
        data_names = []
        for i in range(start_i,start_i + batchsize,1):
            file_path = prefix + audio_paths_list[i]
            file_path = file_path[:-1] # remove /n
            print(str(i) + ': ' + file_path)
            if first:
                data = compute_spectrogram(file_path,config['sampling_rate'])
                first = False
            else:
                data = np.append(data,compute_spectrogram(file_path,config['sampling_rate']), axis=0)
            ground_truth.append(datasets.path2gt(file_path, config['dataset']))
            data_names.append(file_path)
        yield data, ground_truth, data_names

    if leftover:
        first = True
        ground_truth = []
        data_names = []
        for i in range(start_i + batchsize, start_i + batchsize + n_leftover,1):
            file_path = prefix + audio_paths_list[i]
            file_path = file_path[:-1] # remove /n
            print(str(i) + ': ' + file_path)
            if first:
                data = compute_spectrogram(file_path,config['sampling_rate'])
                first = False
            else:
                data = np.append(data,compute_spectrogram(file_path,config['sampling_rate']), axis=0)
            ground_truth.append(datasets.path2gt(file_path, config['dataset']))
            data_names.append(file_path)
        yield data, ground_truth, data_names


def format_cnn_data(prefix, list_audios):
    l_audios = open(list_audios, 'r')
    audio_paths_list = []
    for s in l_audios:
        audio_paths_list.append(s)
    X = []
    Y = []
    ID = []
    for batch in iterate_minibatches(prefix, audio_paths_list, config['CNN']['batch_size']):      
        # feature_maps[i][j, k, l, m]
        # i: layer where extracted the feature
        # j: batch-sample dimension
        # k: one feature-map axis
        # l: other feature-map axis
        # m: feature-map
        feature_maps = sess.run(features_definition, feed_dict={x_in: batch[0]})
        for j in range(batch[0].shape[0]): # for every song in a batch
            tmp_features = np.zeros((len(feature_maps),feature_maps[0].shape[-1]))
            for i in range(len(feature_maps)): # for every bunch of extracted features
                for m in range(feature_maps[i].shape[-1]): # for every feature-map
                    if len(feature_maps[i].shape) == 3:
                        tmp_features[i, m] = np.mean(np.squeeze(feature_maps[i][j, :, m]))
                    elif len(feature_maps[i].shape) == 4:
                        tmp_features[i, m] = np.mean(np.squeeze(feature_maps[i][j, :, :, m]))
            X.append(tmp_features)
            Y.append(batch[1][j]) 
            ID.append(batch[2][j])
        print(Y)
        print(np.array(X).shape)
        print(np.array(ID).shape)

    return X, Y, ID


def compute_spectrogram(audio_path, sampling_rate):
    try:
        # compute spectrogram
        audio, sr = librosa.load(audio_path, sr=sampling_rate)
        audio_rep = librosa.feature.melspectrogram(y=audio,
                                               sr=sampling_rate,
                                               hop_length=256,
                                               n_fft=512,
                                               n_mels=config['CNN']['n_mels'],
                                               power=2,
                                               fmin=0.0,
                                               fmax=6000.0).T

        # normalize audio representation
        print(audio_rep.shape)
        src = librosa.core.logamplitude(audio_rep)

        # zero-pad, repeat-pad and corpping are different in CNNs for having fixed-lengths patches in CNNs
        if config['fix_length_by'] == 'zero-pad' and len(src) < config['CNN']['n_frames']:
            print('Zero padding!')
            src_zeros = np.zeros((config['CNN']['n_frames'],config['CNN']['n_mels']))
            src_zeros[:len(src)] = src
            src = src_zeros

        elif config['fix_length_by'] == 'repeat-pad' and len(src) < config['CNN']['n_frames']:
            print('Repeat and crop to the fixed_length!')
            src_repeat = src
            while (src_repeat.shape[0] < config['CNN']['n_frames']):
                src_repeat = np.concatenate((src_repeat, src), axis=0)    
            src = src_repeat
            src = src[:config['CNN']['n_frames'], :]

        elif config['fix_length_by'] == 'crop':
            print('Cropping audio!')
            src = src[:config['CNN']['n_frames'], :]

        audio_rep = np.expand_dims(src, axis=0) # let the tensor be
        return audio_rep
    except:
        import ipdb; ipdb.set_trace()

def cnn_small_filters():

    with tf.name_scope('cnn_as_choi'):
        global x_in
        x_in = tf.placeholder(tf.float32, [None, None, config['CNN']['n_mels']])

        print('Input: ' + str(x_in.get_shape))

        input_layer = tf.reshape(x_in,[-1, config['CNN']['n_frames'], config['CNN']['n_mels'], 1])
        conv1 = tf.layers.conv2d(inputs=input_layer,
                                 filters=config['CNN']['num_filters'],
                                 kernel_size=[3, 3],
                                 padding='same',
                                 activation=tf.nn.elu,
                                 name='1CNN',
                                 kernel_initializer=tf.contrib.layers.variance_scaling_initializer())
        pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[4, 2], strides=[4, 2])

        conv2 = tf.layers.conv2d(inputs=pool1,
                                 filters=config['CNN']['num_filters'],
                                 kernel_size=[3, 3],
                                 padding='same',
                                 activation=tf.nn.elu,
                                 name='2CNN',
                                 kernel_initializer=tf.contrib.layers.variance_scaling_initializer())
        pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[4, 3], strides=[4, 3])

        conv3 = tf.layers.conv2d(inputs=pool2,
                                 filters=config['CNN']['num_filters'],
                                 kernel_size=[3, 3],
                                 padding='same',
                                 activation=tf.nn.elu,
                                 name='3CNN',
                                 kernel_initializer=tf.contrib.layers.variance_scaling_initializer())
        pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[5, 2], strides=[5, 2])

        conv4 = tf.layers.conv2d(inputs=pool3,
                                 filters=config['CNN']['num_filters'],
                                 kernel_size=[3, 3],
                                 padding='same',
                                 activation=tf.nn.elu,
                                 name='4CNN',
                                 kernel_initializer=tf.contrib.layers.variance_scaling_initializer())
        pool4 = tf.layers.max_pooling2d(inputs=conv4, pool_size=[4, 2], strides=[4, 2])

        conv5 = tf.layers.conv2d(inputs=pool4, 
                                 filters=config['CNN']['num_filters'], 
                                 kernel_size=[3, 3], 
                                 padding='same', 
                                 activation=tf.nn.elu,
                                 name='5CNN', 
                                 kernel_initializer=tf.contrib.layers.variance_scaling_initializer())
        pool5 = tf.layers.max_pooling2d(inputs=conv5, pool_size=[4, 4], strides=[4, 4])

    global sess
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())

    print(pool1.get_shape)
    print(pool2.get_shape)
    print(pool3.get_shape)
    print(pool4.get_shape)
    print(pool5.get_shape)

    return [pool1, pool2, pool3, pool4, pool5]


def cnn_music():
   
    # remove some temporal filters to have the same ammount of timbral and temporal filters
    if config['CNN']['num_filters'] == 256:
        remove = 64  
    elif config['CNN']['num_filters'] == 128:
        remove = 32    
    elif config['CNN']['num_filters'] == 64:
        remove = 16
    elif config['CNN']['num_filters'] == 32:
        remove = 8
    elif config['CNN']['num_filters'] == 16:
        remove = 4
    elif config['CNN']['num_filters'] == 8:
        remove = 2
    elif config['CNN']['num_filters'] == 4:
        remove = 1

    # define the cnn_music model  
    with tf.name_scope('cnn_music'):
        global x_in
        x_in = tf.placeholder(tf.float32, [None, None, config['CNN']['n_mels']])

        print('Input: ' + str(x_in.get_shape))

        input_layer = tf.reshape(x_in, [-1, config['CNN']['n_frames'], config['CNN']['n_mels'], 1])

        # padding only time domain for an efficient 'same' implementation
        # (since we pool throughout all frequency afterwards)
        input_pad_7 = tf.pad(input_layer, [[0, 0], [3, 3], [0, 0], [0, 0]], "CONSTANT")
        input_pad_3 = tf.pad(input_layer, [[0, 0], [1, 1], [0, 0], [0, 0]], "CONSTANT")

        # [TIMBRE] filter shape 1: 7x0.9f
        conv1 = tf.layers.conv2d(inputs=input_pad_7,
                             filters=config['CNN']['num_filters'],
                             kernel_size=[7, int(0.9 * config['CNN']['n_mels'])],
                             padding="valid",
                             activation=tf.nn.relu,
                             kernel_initializer=tf.contrib.layers.variance_scaling_initializer())
        pool1 = tf.layers.max_pooling2d(inputs=conv1,
                                    pool_size=[1, conv1.shape[2]],
                                    strides=[1, conv1.shape[2]])
        p1 = tf.squeeze(pool1, [2])

        # [TIMBRE] filter shape 2: 3x0.9f
        conv2 = tf.layers.conv2d(inputs=input_pad_3, 
                             filters=config['CNN']['num_filters']*2,
                             kernel_size=[3, int(0.9 * config['CNN']['n_mels'])],
                             padding="valid",
                             activation=tf.nn.relu,
                             kernel_initializer=tf.contrib.layers.variance_scaling_initializer())
        pool2 = tf.layers.max_pooling2d(inputs=conv2,
                                    pool_size=[1, conv2.shape[2]],
                                    strides=[1, conv2.shape[2]])
        p2 = tf.squeeze(pool2, [2])

        # [TIMBRE] filter shape 3: 1x0.9f
        conv3 = tf.layers.conv2d(inputs=input_layer, 
                             filters=config['CNN']['num_filters']*4,
                             kernel_size=[1, int(0.9 * config['CNN']['n_mels'])],
                             padding="valid",
                             activation=tf.nn.relu,
                             kernel_initializer=tf.contrib.layers.variance_scaling_initializer())
        pool3 = tf.layers.max_pooling2d(inputs=conv3,
                                    pool_size=[1, conv3.shape[2]],
                                    strides=[1, conv3.shape[2]])
        p3 = tf.squeeze(pool3, [2])

        # [TIMBRE] filter shape 3: 7x0.4f
        conv4 = tf.layers.conv2d(inputs=input_pad_7,
                             filters=config['CNN']['num_filters'],
                             kernel_size=[7, int(0.4 * config['CNN']['n_mels'])],
                             padding="valid",
                             activation=tf.nn.relu,
                             kernel_initializer=tf.contrib.layers.variance_scaling_initializer())
        pool4 = tf.layers.max_pooling2d(inputs=conv4,
                                    pool_size=[1, conv4.shape[2]],
                                    strides=[1, conv4.shape[2]])
        p4 = tf.squeeze(pool4, [2])

        # [TIMBRE] filter shape 5: 3x0.4f
        conv5 = tf.layers.conv2d(inputs=input_pad_3, 
                             filters=config['CNN']['num_filters']*2,
                             kernel_size=[3, int(0.4 * config['CNN']['n_mels'])],
                             padding="valid",
                             activation=tf.nn.relu,
                             kernel_initializer=tf.contrib.layers.variance_scaling_initializer())
        pool5 = tf.layers.max_pooling2d(inputs=conv5, pool_size=[1, conv5.shape[2]],
                                    strides=[1, conv5.shape[2]])
        p5 = tf.squeeze(pool5, [2])

        # [TIMBRE] filter shape 6: 1x0.4f
        conv6 = tf.layers.conv2d(inputs=input_layer, 
                             filters=config['CNN']['num_filters']*4,
                             kernel_size=[1, int(0.4 * config['CNN']['n_mels'])], padding="valid",
                             activation=tf.nn.relu,
                             kernel_initializer=tf.contrib.layers.variance_scaling_initializer())
        pool6 = tf.layers.max_pooling2d(inputs=conv6, pool_size=[1, conv6.shape[2]],
                                    strides=[1, conv6.shape[2]])
        p6 = tf.squeeze(pool6, [2])

        # [TEMPORAL-FEATURES] - average pooling + filter shape 7: 165x1
        pool7 = tf.layers.average_pooling2d(inputs=input_layer,
                                        pool_size=[1, config['CNN']['n_mels']],
                                        strides=[1, config['CNN']['n_mels']])
        pool7_rs = tf.squeeze(pool7, [3])
        conv7 = tf.layers.conv1d(inputs=pool7_rs,
                             filters=config['CNN']['num_filters']-remove,
                             kernel_size=165,
                             padding="same",
                             activation=tf.nn.relu,
                             kernel_initializer=tf.contrib.layers.variance_scaling_initializer())

        # [TEMPORAL-FEATURES] - average pooling + filter shape 8: 128x1
        pool8 = tf.layers.average_pooling2d(inputs=input_layer,
                                        pool_size=[1, config['CNN']['n_mels']],
                                        strides=[1, config['CNN']['n_mels']])
        pool8_rs = tf.squeeze(pool8, [3])
        conv8 = tf.layers.conv1d(inputs=pool8_rs,
                             filters=config['CNN']['num_filters']*2-remove,
                             kernel_size=128,
                             padding="same",
                             activation=tf.nn.relu,
                             kernel_initializer=tf.contrib.layers.variance_scaling_initializer())

        # [TEMPORAL-FEATURES] - average pooling + filter shape 9: 64x1
        pool9 = tf.layers.average_pooling2d(inputs=input_layer,
                                        pool_size=[1, config['CNN']['n_mels']],
                                        strides=[1, config['CNN']['n_mels']])
        pool9_rs = tf.squeeze(pool9, [3])
        conv9 = tf.layers.conv1d(inputs=pool9_rs,
                             filters=config['CNN']['num_filters']*4-remove,
                             kernel_size=64,
                             padding="same",
                             activation=tf.nn.relu,
                             kernel_initializer=tf.contrib.layers.variance_scaling_initializer())

        # [TEMPORAL-FEATURES] - average pooling + filter shape 10: 32x1
        pool10 = tf.layers.average_pooling2d(inputs=input_layer,
                                         pool_size=[1, config['CNN']['n_mels']],
                                         strides=[1, config['CNN']['n_mels']])
        pool10_rs = tf.squeeze(pool10, [3])
        conv10 = tf.layers.conv1d(inputs=pool10_rs,
                              filters=config['CNN']['num_filters']*8-remove,
                              kernel_size=32,
                              padding="same",
                              activation=tf.nn.relu,
                              kernel_initializer=tf.contrib.layers.variance_scaling_initializer())

    global sess
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())

    # concatenate all feature maps
    timbral = tf.concat([p1, p2, p3, p4, p5, p6], 2)
    temporal = tf.concat([conv7, conv8, conv9, conv10], 2)

    print(timbral.get_shape)
    print(temporal.get_shape)

    # check [moving_mean, moving_variance, beta, gamma]
    #    [batch_normalization_8/moving_mean:0, batch_normalization_8/moving_variance:0,
    #    .. 'batch_normalization_8/beta', 'batch_normalization_8/gamma:0']
    #    sess.run('batch_normalization_8/moving_mean:0')

    return [timbral, temporal]


def select_cnn_feature_layers(feature_maps, selected_features_list):
    selected_features = []
    for i in range(len(feature_maps)):
        tmp = np.array([])
        for j in selected_features_list:
            tmp = np.concatenate((tmp, np.squeeze(feature_maps[i][j])))
        selected_features.append(tmp)
    return selected_features


# MFCCs

def extract_mfcc_features(audio, sampling_rate=12000): 
    src, sr = librosa.load(audio, sr=sampling_rate)

    # zero-pad, repeat-pad and corpping are different in CNNs for having fixed-lengths patches in CNNs
    if config['fix_length_by'] == 'zero-pad' and len(src) < config['MFCC']['fixed_length']:
        print('Zero padding!')
        import time
        time.sleep(60)
        src_zeros = np.zeros(config['MFCC']['fixed_length']) # min length to have 3-frame mfcc's
        src_zeros[:len(src)] = src
        src = src_zeros
    elif config['fix_length_by'] == 'repeat-pad' and len(src) < config['MFCC']['fixed_length']:
        print('Repeat padding!')
        import time
        time.sleep(60)
        src_repeat = src
        while (len(src_repeat) < config['MFCC']['fixed_length']):
            src_repeat = np.concatenate((src_repeat, src), axis=0)    
        src = src_repeat
    elif config['fix_length_by'] == 'crop':
        print('Copping audio!')
        import time
        time.sleep(60)
        src = src[:config['MFCC']['fixed_length']]

    print(len(src))
    # dmfcc as in https://github.com/keunwoochoi/transfer_learning_music/
    mfcc = librosa.feature.mfcc(src, sampling_rate, n_mfcc=config['MFCC']['number'])
    dmfcc = mfcc[:, 1:] - mfcc[:, :-1]
    ddmfcc = dmfcc[:, 1:] - dmfcc[:, :-1]
    return np.concatenate((np.mean(mfcc, axis=1), np.std(mfcc, axis=1),
                           np.mean(dmfcc, axis=1), np.std(dmfcc, axis=1),
                           np.mean(ddmfcc, axis=1), np.std(ddmfcc, axis=1)), 
                           axis=0)


def format_mfcc_data(prefix, list_audios):
    songs_list = open(list_audios, 'r')
    X = []
    Y = []
    ID = []
    n_song = 0
    for song in songs_list:
        ground_truth = song[:song.rfind('/')] # remove?
        print(str(n_song) + ': ' + song[:-1])
        X.append(extract_mfcc_features(prefix + song[:-1], config['sampling_rate']))
        #import ipdb; ipdb.set_trace()
        Y.append(datasets.path2gt(song[:-1], config['dataset']))
        ID.append(song[:-1])
        n_song += 1
        print(Y)
        print(np.array(X).shape)
    return X, Y, ID


if __name__ == '__main__':

    #--------------------#
    # FEATURE EXTRACTION #
    #--------------------#
    
    print(config)

    if not config['load_extracted_features']: 

        print('Set file name (unique identifier) for the experiment..')
        if config['features_type'] == 'MFCC':
            experiment_name = str(config['experiment_name']) + '_MFCC_' + str(int(time.time()))
        elif config['features_type'] == 'CNN':
            experiment_name = str(config['experiment_name']) + '_CNN_' + str(config['CNN']['n_mels']) \
                + '_' + str(config['CNN']['n_frames']) + '_' + str(config['CNN']['batch_size']) \
                + '_' + str(config['CNN']['architecture']) + '_' + str(config['CNN']['num_filters']) \
                + '_' + str(config['CNN']['selected_features_list']) + '_'+ str(int(time.time()))
        print(experiment_name)


        print('Extracting features..')
        if config['features_type'] == 'CNN':
            if config['CNN']['architecture'] == 'cnn_small_filters':
                features_definition = cnn_small_filters()
            elif config['CNN']['architecture'] == 'cnn_music':
                features_definition = cnn_music()
            print('Number of parameters of the model: ' + str(
                   count_params(tf.trainable_variables()))+'\n')
            if config['audios_list'] == False:
                print('Extract features for train-set..')
                x_train, y_train, id_train = format_cnn_data(prefix=config['audio_path'],
                                     list_audios=config['train_set_list'])
                print('Extract features for val-set..')
                x_val, y_val, id_val = format_cnn_data(prefix=config['audio_path'],
                                 list_audios=config['val_set_list'])
                print('Extract features for test-set..')
                x_test, y_test, id_test = format_cnn_data(prefix=config['audio_path'],
                                   list_audios=config['test_set_list'])
            else:
                x, y, ids = format_cnn_data(prefix=config['audio_path'],
                         list_audios=config['audios_list'])

        elif config['features_type'] == 'MFCC':
            if config['audios_list'] == False:
                print('Extract features for train-set..')
                x_train, y_train, id_train = format_mfcc_data(prefix=config['audio_path'],
                                     list_audios=config['train_set_list'])
                print('Extract features for val-set..')
                x_val, y_val, id_val = format_mfcc_data(prefix=config['audio_path'],
                                 list_audios=config['val_set_list'])
                print('Extract features for test-set..')
                x_test, y_test, id_test = format_mfcc_data(prefix=config['audio_path'],
                                   list_audios=config['test_set_list'])
            else:
                x, y, ids = format_mfcc_data(prefix=config['audio_path'],
                         list_audios=config['audios_list'])


        print('Storing extracted features..')        
        if not os.path.exists(config['save_extracted_features_folder']):
            os.makedirs(config['save_extracted_features_folder'])

        if config['audios_list'] == False:
            with open(config['save_extracted_features_folder'] + experiment_name + '.pkl', 'wb') as f:
                pickle.dump([x_train, y_train, id_train, x_val, y_val, id_val, x_test, y_test, id_test, config], f)
        else:
            with open(config['save_extracted_features_folder'] + experiment_name + '.pkl', 'wb') as f:
                pickle.dump([x, y, ids, config], f)            

    else:  # load extracted features
        
        print('Loading features: ' + config['load_extracted_features'])

        if config['audios_list'] == False:
            with open(config['load_extracted_features'], 'rb') as f:
                x_train, y_train, id_train, x_val, y_val, id_val, x_test, y_test, id_test, config = pickle.load(f)
        else:
            with open(config['load_extracted_features'], 'rb') as f:
                x, y, ids, config = pickle.load(f)

    if config['features_type'] == 'CNN':

        print('Select CNN features..')

        print('Data size (data points, feature vector)..')
        if config['audios_list'] == False:
            x_train = select_cnn_feature_layers(x_train, config['CNN']['selected_features_list'])
            x_val = select_cnn_feature_layers(x_val, config['CNN']['selected_features_list'])
            x_test = select_cnn_feature_layers(x_test, config['CNN']['selected_features_list'])
            print(np.array(x_train).shape)
            print(np.array(x_val).shape)
            print(np.array(x_test).shape)
        else:
            x = select_cnn_feature_layers(x, config['CNN']['selected_features_list'])
            print(np.array(x).shape)

    #------------#
    # CLASSIFIER #
    #------------#

    if not os.path.exists(config['results_folder']):
        os.makedirs(config['results_folder'])
    #f = open(config['results_folder'] + experiment_name + '.txt','w')
    f = open(config['results_folder'] + 'moguda.txt','w')
    if config['dataset'] == 'UrbanSound8K':
        
        #trobar putes partitions
        print('Oju!')
        #import ipdb; ipdb.set_trace()
        #import time
        #time.sleep(60)

        df = pd.read_csv('/data/UrbanSound8K/metadata/UrbanSound8K.csv')
        folds_mask = []
        for i in ids:
            tag = i[i.rfind('/')+1:]
            folds_mask.append(int(df[df.slice_file_name==tag].fold))
        ps = PredefinedSplit(test_fold=folds_mask)

        # IMPRIMR PS PER VEURE QUE PASSA!!! IPDB AQUI?
        #import ipdb; ipdb.set_trace()

        print('Peligru!')
        svc = SVC()
        svm = GridSearchCV(svc, svm_params, cv=ps, n_jobs=3, pre_dispatch=3*8).fit(x, y)

        print('Best score of {}: {}'.format(svm.best_score_,svm.best_params_))
        print(svm.best_score_)
        print(config)

        print('Storing results..')        

        f.write('Best score of ' + str(svm.best_score_) + ': ' + str(svm.best_params_))
        f.write(str(config))



    elif config['audios_list'] == False:
        x_dev = np.concatenate((x_train, x_val), axis=0)
        y_dev = np.concatenate((y_train, y_val), axis=0)
        val_mask = np.concatenate((-np.ones(len(y_train)), np.zeros(len(y_val))), axis=0)
        ps = PredefinedSplit(test_fold=val_mask)
        svc = SVC()
        hps = GridSearchCV(svc, svm_params, cv=ps, n_jobs=3, pre_dispatch=3*8).fit(x_dev, y_dev)
        svm = SVC()
        svm.set_params(**hps.best_params_)
        svm.fit(x_train, y_train)
        y_true, y_pred = y_test, svm.predict(x_test)
        print('Detailed classification report:')
        print(classification_report(y_true, y_pred))
        print('Accuracy test set: ')
        print(accuracy_score(y_test, svm.predict(x_test)))
        print(config)

        print('Storing results..')   
        f.write(str(classification_report(y_true, y_pred)))     
        f.write('Accuracy: ' + str(accuracy_score(y_test, svm.predict(x_test))) + '\n')
        f.write(str(config))

    else:
        svc = SVC()
        if config['debug']:
            svm = GridSearchCV(svc, svm_params, cv=2, n_jobs=3, pre_dispatch=3*8).fit(x, y)
            print('[DEBUG MODE] 2 fold cross-validation!')
        else:
            svm = GridSearchCV(svc, svm_params, cv=10, n_jobs=3, pre_dispatch=3*8).fit(x, y)
            print('10 fold cross-validation!')

        print('Best score of {}: {}'.format(svm.best_score_,svm.best_params_))
        print(svm.best_score_)
        print(config)

        print('Storing results..')        

        f.write('Best score of ' + str(svm.best_score_) + ': ' + str(svm.best_params_))
        f.write(str(config))

    f.close()

# NOTES ON SPECTROGRAM. Mel power spectrogram. Sampling rate: 12k. fmin=0 and fmax=6000. Using shorter clips.

# IDEAS. Check statistics of input data (zero-mean/one-var)?
#      . Only store mean values for features?
