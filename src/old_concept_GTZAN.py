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

config = {
    'experiment_name': 'dev_num_features',
    'pre_processing': {
        'n_mels': 96,  # number of mel bands of the input time-freq representation
        'n_frames': 1360  # number of time-frames of hte input time-freq representation
    },
    'features_type': 'noCNN',
    'selected_features_list': [0, 1, 2, 3, 4],
    'load_extracted_features': '/mnt/vmdata/users/jpons/elmarc/data/GTZAN/features/MFCCs_1360_961513099930.pkl',
    # load already extracted features by defining the
    # path where these are saved - set as False for extracting those again
    'audio_path': '/homedtic/jpons/GTZAN/',
    'train_set_list': '/homedtic/jpons/GTZAN_partitions/train_filtered.txt',
    'val_set_list': '/homedtic/jpons/GTZAN_partitions/valid_filtered.txt',
    'test_set_list': '/homedtic/jpons/GTZAN_partitions/test_filtered.txt',
    'save_extracted_features_folder': '../data/GTZAN/features/',
    'partition': 'random'
}

# svm_params = [
#     {'kernel': ['rbf', 'poly', 'sigmoid'],
#      'gamma': [1 / (2 ** 3), 1 / (2 ** 5), 1 / (2 ** 7), 1 / (2 ** 9), 1 / (2 ** 11), 'auto'],
#      'C': [0.1, 2, 8, 32, 100, 1000, 10000, 100000]},
#
#     {'kernel': ['linear'],
#      'C': [0.1, 2, 8, 32, 100, 1000, 10000, 100000]}
# ]

svm_params = [
    {'kernel': ['rbf'],
     'gamma': [1 / (2 ** 3), 1 / (2 ** 5), 1 / (2 ** 7), 1 / (2 ** 9), 1 / (2 ** 11), 'auto'],
     'C': [0.1, 2, 8, 32, 100, 1000, 10000, 100000]}
]


def select_cnn_feature_layers(feature_maps, selected_features_list):
    selected_features = []
    for i in range(len(feature_maps)):
        for j in selected_features_list:
            selected_features.append(np.squeeze(feature_maps[i][j]))
    return selected_features


def extract_features(audio, feature_type, sampling_rate=16000):
    """
    Extract tensor-flow features: extract audio, compute librosa features and
    pass it through the tensor-flow model to extract the *features_list*

    :param audio: String pointing where the audio is located
    :param sampling_rate: Sampling rate used when loading the audio (change it for down-sampling)

    :return features: Extracted features per *audio* song
    """
    if feature_type == 'MFCC':
        audio, sr = librosa.load(audio, sr=sampling_rate)

        mfcc = librosa.feature.mfcc(y=audio, sr=sampling_rate, n_mfcc=20)
        mfcc_mean = np.mean(mfcc, axis=1)
        mfcc_std = np.std(mfcc, axis=1)

        mfcc_delta = librosa.feature.delta(mfcc)
        mfcc_delta_mean = np.mean(mfcc_delta, axis=1)
        mfcc_delta_std = np.std(mfcc_delta, axis=1)

        mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
        mfcc_delta2_mean = np.mean(mfcc_delta2, axis=1)
        mfcc_delta2_std = np.std(mfcc_delta2, axis=1)

        return np.concatenate((mfcc_mean, mfcc_std,
                               mfcc_delta_mean, mfcc_delta_std,
                               mfcc_delta2_mean, mfcc_delta2_std), axis=0)

    elif feature_type == 'CNN':
        # compute spectrogram
        audio, sr = librosa.load(audio, sr=sampling_rate)
        audio_rep = librosa.feature.melspectrogram(y=audio,
                                                   sr=sampling_rate,
                                                   hop_length=256,
                                                   n_fft=512,
                                                   n_mels=config['pre_processing']['n_mels']).T

        # normalize audio representation
        audio_rep = np.log10(10000 * audio_rep + 1)
        # audio_rep = (audio_rep - config['patches_params']['mean']) / config['patches_params']['std']
        audio_rep = np.expand_dims(audio_rep, axis=0)
        audio_rep = audio_rep[:, :config['pre_processing']['n_frames'], :]

        # extract features
        feature_maps = sess.run(features_definition, feed_dict={x: audio_rep, is_train: False})
        features = []
        for i in range(len(feature_maps)):
            tmp_features = []
            for j in range(feature_maps[i].shape[3]):
                tmp_features.append(np.mean(np.squeeze(feature_maps[i][:, :, :, j])))
            features.append(tmp_features)
    return features


def format_data_gtzan(prefix, list_audios, features_type):
    """
    Extract features and attach its label to every song

    :param prefix: String pointing the root of the folder where the audios are stored
    :param list_audios: String pointing a .txt file where all the audios of the partition are listed

    :return X: Extracted features per song
    :return Y: Label attached to each song
    """
    songs_list = open(list_audios, 'r')
    X = []
    Y = []
    n_song = 0
    for song in songs_list:
        ground_truth = song[:song.rfind('/')]
        print(str(n_song) + ': ' + song[:-1])
        if ground_truth == 'blues':
            X.append(extract_features(prefix + song[:-1], features_type))
            Y.append(0)
        elif ground_truth == 'classical':
            X.append(extract_features(prefix + song[:-1], features_type))
            Y.append(1)
        elif ground_truth == 'country':
            X.append(extract_features(prefix + song[:-1], features_type))
            Y.append(2)
        elif ground_truth == 'disco':
            X.append(extract_features(prefix + song[:-1], features_type))
            Y.append(3)
        elif ground_truth == 'hiphop':
            X.append(extract_features(prefix + song[:-1], features_type))
            Y.append(4)
        elif ground_truth == 'jazz':
            X.append(extract_features(prefix + song[:-1], features_type))
            Y.append(5)
        elif ground_truth == 'metal':
            X.append(extract_features(prefix + song[:-1], features_type))
            Y.append(6)
        elif ground_truth == 'pop':
            X.append(extract_features(prefix + song[:-1], features_type))
            Y.append(7)
        elif ground_truth == 'reggae':
            X.append(extract_features(prefix + song[:-1], features_type))
            Y.append(8)
        elif ground_truth == 'rock':
            X.append(extract_features(prefix + song[:-1], features_type))
            Y.append(9)
        else:
            print('Warning: did not find the corresponding ground truth (' + str(ground_truth) + ').')
        n_song += 1
        print(Y)
        print(np.array(X).shape)
    return X, Y


def model():
    with tf.name_scope('model'):
        global x
        x = tf.placeholder(tf.float32, [None, None, config['pre_processing']['n_mels']])

        global is_train
        is_train = tf.placeholder(tf.bool)

        print('Input: ' + str(x.get_shape))

        bn_x = tf.layers.batch_normalization(x, training=is_train)
        input_layer = tf.reshape(bn_x,
                                 [-1, config['pre_processing']['n_frames'], config['pre_processing']['n_mels'], 1])
        conv1 = tf.layers.conv2d(inputs=input_layer,
                                 filters=32,
                                 kernel_size=[3, 3],
                                 padding='valid',
                                 activation=tf.nn.elu,
                                 name='1CNN',
                                 kernel_initializer=tf.contrib.layers.variance_scaling_initializer())
        bn_conv1 = tf.layers.batch_normalization(conv1, training=is_train)
        dropped_conv1 = tf.layers.dropout(inputs=bn_conv1, rate=0.3, training=is_train)
        pool1 = tf.layers.max_pooling2d(inputs=dropped_conv1, pool_size=[4, 2], strides=[4, 2])

        conv2 = tf.layers.conv2d(inputs=pool1,
                                 filters=32,
                                 kernel_size=[3, 3],
                                 padding='valid',
                                 activation=tf.nn.elu,
                                 name='2CNN',
                                 kernel_initializer=tf.contrib.layers.variance_scaling_initializer())
        bn_conv2 = tf.layers.batch_normalization(conv2, training=is_train)
        dropped_conv2 = tf.layers.dropout(bn_conv2, rate=0.3, training=is_train)
        pool2 = tf.layers.max_pooling2d(inputs=dropped_conv2, pool_size=[4, 3], strides=[4, 3])

        conv3 = tf.layers.conv2d(inputs=pool2,
                                 filters=32,
                                 kernel_size=[3, 3],
                                 padding='valid',
                                 activation=tf.nn.elu,
                                 name='3CNN',
                                 kernel_initializer=tf.contrib.layers.variance_scaling_initializer())
        bn_conv3 = tf.layers.batch_normalization(conv3, training=is_train)
        dropped_conv3 = tf.layers.dropout(bn_conv3, rate=0.3, training=is_train)
        pool3 = tf.layers.max_pooling2d(inputs=dropped_conv3, pool_size=[5, 1], strides=[5, 1])

        conv4 = tf.layers.conv2d(inputs=pool3,
                                 filters=32,
                                 kernel_size=[3, 3],
                                 padding='valid',
                                 activation=tf.nn.elu,
                                 name='4CNN',
                                 kernel_initializer=tf.contrib.layers.variance_scaling_initializer())
        bn_conv4 = tf.layers.batch_normalization(conv4, training=is_train)
        dropped_conv4 = tf.layers.dropout(bn_conv4, rate=0.3, training=is_train)
        pool4 = tf.layers.max_pooling2d(inputs=dropped_conv4, pool_size=[4, 3], strides=[4, 3])

        conv5 = tf.layers.conv2d(inputs=pool4, filters=32, kernel_size=[3, 3], padding='valid', activation=tf.nn.elu,
                                 name='5CNN', kernel_initializer=tf.contrib.layers.variance_scaling_initializer())

    global sess
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())

    return [conv1, conv2, conv3, conv4, conv5]


if __name__ == '__main__':

    # FEATURE EXTRACTION

    if not config['load_extracted_features']:  # extract and store features

        if config['features_type'] == 'CNN':
            features_definition = model()

        features_path = str(config['experiment_name']) + '_' + str(config['pre_processing']['n_frames']) + \
                        '_' + str(str(config['pre_processing']['n_mels'])) + str(int(time.time()))

        print('Extract features for train-set..')
        x_pre_train, y_train = format_data_gtzan(prefix=config['audio_path'],
                                                 list_audios=config['train_set_list'],
                                                 features_type=config['features_type'])

        print('Extract features for val-set..')
        x_pre_val, y_val = format_data_gtzan(prefix=config['audio_path'],
                                             list_audios=config['val_set_list'],
                                             features_type=config['features_type'])

        print('Extract features for test-set..')
        x_pre_test, y_test = format_data_gtzan(prefix=config['audio_path'],
                                               list_audios=config['test_set_list'],
                                               features_type=config['features_type'])

        # storing extacted features
        if not os.path.exists(config['save_extracted_features_folder']):
            os.makedirs(config['save_extracted_features_folder'])
        with open(config['save_extracted_features_folder'] + features_path + '.pkl', 'wb') as f:
            pickle.dump([x_pre_train, y_train, x_pre_val, y_val, x_pre_test, y_test], f)

    else:  # load extracted features
        print('Loading features: ' + config['load_extracted_features'])
        with open(config['load_extracted_features'], 'rb') as f:
            x_pre_train, y_train, x_pre_val, y_val, x_pre_test, y_test = pickle.load(f)

    if config['features_type'] == 'CNN':
        # select features
        print('in CNN')
        x_train = select_cnn_feature_layers(x_pre_train, config['selected_features_list'])
        x_val = select_cnn_feature_layers(x_pre_val, config['selected_features_list'])
        x_test = select_cnn_feature_layers(x_pre_test, config['selected_features_list'])
    else:
        x_train = x_pre_train
        x_val = x_pre_val
        x_test = x_pre_test

    print(np.array([x_train]).shape)
    print(np.array([x_val]).shape)
    print(np.array([x_test]).shape)

    # CLASSIFIER

    if config['partition'] == 'defined':
        x_dev = np.concatenate((x_train, x_val), axis=0)
        y_dev = np.concatenate((y_train, y_val), axis=0)
        val_mask = np.concatenate((-np.ones(len(y_train)), np.zeros(len(y_val))), axis=0)
        ps = PredefinedSplit(test_fold=val_mask)
        svc = SVC()
        svm_hps = GridSearchCV(svc, svm_params, scoring='accuracy', cv=ps)
        svm_hps.fit(x_dev, y_dev)
        print('Detailed classification report 0:')
        svm_test = SVC()
        svm_test.set_params(**svm_hps.best_params_)
        print(svm_test)
        svm_test.fit(x_train, y_train)
        y_true, y_pred = y_test, svm_test.predict(x_test)
        print(classification_report(y_true, y_pred))
        print('Accuracy test set: ')
        print(accuracy_score(y_test, svm_test.predict(x_test)))
    else:
        x_dev = np.concatenate((x_train, x_val, x_test), axis=0)
        y_dev = np.concatenate((y_train, y_val, y_test), axis=0)
        svc = SVC()
        svm_hps = GridSearchCV(svc, svm_params, cv=10, n_jobs=3, pre_dispatch=3*8).fit(x_dev, y_dev)
        print('best score of {}: {}'.format(svm_hps.best_score_,svm_hps.best_params_))
        print('Detailed classification report 1:')
        #import ipdb; ipdb.set_trace()
        print(svm_hps.best_score_)

# COMMENT 1: with linear kernel as option, results get much worse in test-set.
# BEST RESULT MFCC: 0.524, with rbf kernel only.
# BEST RESULT BN-CNN: 0.431, with rbf kernel only.

