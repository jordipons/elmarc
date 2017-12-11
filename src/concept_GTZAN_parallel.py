import numpy as np
from sklearn.svm import SVC
import tensorflow as tf
import librosa
import pickle
import time
from joblib import Parallel, delayed

config = {
    'model_name': '3x3_random',
    'pre_processing': {
        'n_mel_bands': 96,  # number of mel bands of the input time-freq representation
        'n_frames': 1360  # number of time-frames of hte input time-freq representation
    },
    'load_extracted_features': False, # load already extracted features by defining the
    # path where thse are saved - set as False for extracting those again
    'debug': False,
    'num_processing_units': 5,
    'error_code': 99999
}


def extract_cnn_features(audio, sampling_rate=16000):
    '''
    Extract tensor-flow features: extract audio, compute librosa features and
    pass it through the tensor-flow model to extract the *features_list*

    :param audio: String pointing where the audio is located
    :param sampling_rate: Sampling rate used when loading the audio (change it for downsampling)

    :return features: Extracted features per *audio* song
    '''
    # compute spectrogram
    audio, sr = librosa.load(audio, sr=sampling_rate)
    audio_rep = librosa.feature.melspectrogram(y=audio,
                                               sr=sampling_rate,
                                               hop_length=256,
                                               n_fft=512,
                                               n_mels=config['pre_processing']['n_mel_bands']).T

    # normalize audio representation
    audio_rep = np.log10(10000 * audio_rep + 1)
    # audio_rep = (audio_rep - config['patches_params']['mean']) / config['patches_params']['std']
    audio_rep = np.expand_dims(audio_rep, axis=0)
    audio_rep = audio_rep[:, :config['pre_processing']['n_frames'], :]

    # extract features
    feature_maps = sess.run(features_definition, feed_dict={x: audio_rep, is_train: False})
    features = []
    for i in range(len(feature_maps)):
        for j in range(feature_maps[1].shape[3]):
            features.append(np.mean(np.squeeze(feature_maps[i][:, :, :, j])))
    return features


def pairing_data_and_labels_gtzan(prefix, songs, i):
    print(songs[i][:-1])
    X[i] = extract_cnn_features(prefix + songs[i][1:-1])
    ground_truth = songs[i][2:songs[i].rfind('/')]
    print(ground_truth)
    if ground_truth == 'blues':
        Y[i] = 0
    elif ground_truth == 'classical':
        Y[i] = 1
    elif ground_truth == 'country':
        Y[i] = 2
    elif ground_truth == 'disco':
        Y[i] = 3
    elif ground_truth == 'hiphop':
        Y[i] = 4
    elif ground_truth == 'jazz':
        Y[i] = 5
    elif ground_truth == 'metal':
        Y[i] = 6
    elif ground_truth == 'pop':
        Y[i] = 7
    elif ground_truth == 'reggae':
        Y[i] = 8
    elif ground_truth == 'rock':
        Y[i] = 9


def format_data(prefix, list_audios):
    '''
    Extract features and attach its label to every song

    :param prefix: String pointing the root of the folder where the audios are stored
    :param list_audios: String pointing a .txt file where all the audios of the partition are listed

    :return X: Extracted features per song
    :return Y: Label attached to each song
    '''
    list = open(list_audios, 'r')

    n_songs = 0
    songs = []
    for song in list:
        n_songs = n_songs + 1
        songs.append(song)

    global X
    X = np.zeros((n_songs,160))+config['error_code']
    global Y
    Y = np.zeros(n_songs) + config['error_code']

    if config['debug']:
        print('WARNING: Parallelization is not used!')
        for index in range(0, n_songs):
            pairing_data_and_labels_gtzan(prefix, songs, index)

    else:
        Parallel(n_jobs=config['num_processing_units'])(
            delayed(pairing_data_and_labels_gtzan)(prefix, songs, index) for index in range(0, n_songs))

    return X, Y

# DEFINE YOUR MODEL: 3x3_random feature extractor
with tf.name_scope('model'):
    x = tf.placeholder(tf.float32, [None, None, config['pre_processing']['n_mel_bands']])
    is_train = tf.placeholder(tf.bool)

    print('Input: ' + str(x.get_shape))

    input_layer = tf.reshape(x, [-1, config['pre_processing']['n_frames'], config['pre_processing']['n_mel_bands'], 1])
    conv1 = tf.layers.conv2d(inputs=input_layer, filters=32, kernel_size=[3, 3], padding='valid', activation=tf.nn.elu,
                             name='1CNN', kernel_initializer=tf.contrib.layers.variance_scaling_initializer())
    bn_conv1 = tf.layers.batch_normalization(conv1, training=is_train)
    dropped_conv1 = tf.layers.dropout(inputs=bn_conv1, rate=0.3, training=is_train)
    pool1 = tf.layers.max_pooling2d(inputs=dropped_conv1, pool_size=[4, 2], strides=[4, 2])

    conv2 = tf.layers.conv2d(inputs=pool1, filters=32, kernel_size=[3, 3], padding='valid', activation=tf.nn.elu,
                             name='2CNN', kernel_initializer=tf.contrib.layers.variance_scaling_initializer())
    bn_conv2 = tf.layers.batch_normalization(conv2, training=is_train)
    dropped_conv2 = tf.layers.dropout(bn_conv2, rate=0.3, training=is_train)
    pool2 = tf.layers.max_pooling2d(inputs=dropped_conv2, pool_size=[4, 3], strides=[4, 3])

    conv3 = tf.layers.conv2d(inputs=pool2, filters=32, kernel_size=[3, 3], padding='valid', activation=tf.nn.elu,
                             name='3CNN', kernel_initializer=tf.contrib.layers.variance_scaling_initializer())
    bn_conv3 = tf.layers.batch_normalization(conv3, training=is_train)
    dropped_conv3 = tf.layers.dropout(bn_conv3, rate=0.3, training=is_train)
    pool3 = tf.layers.max_pooling2d(inputs=dropped_conv3, pool_size=[5, 1], strides=[5, 1])

    conv4 = tf.layers.conv2d(inputs=pool3, filters=32, kernel_size=[3, 3], padding='valid', activation=tf.nn.elu,
                             name='4CNN', kernel_initializer=tf.contrib.layers.variance_scaling_initializer())
    bn_conv4 = tf.layers.batch_normalization(conv4, training=is_train)
    dropped_conv4 = tf.layers.dropout(bn_conv4, rate=0.3, training=is_train)
    pool4 = tf.layers.max_pooling2d(inputs=dropped_conv4, pool_size=[4, 3], strides=[4, 3])

    conv5 = tf.layers.conv2d(inputs=pool4, filters=32, kernel_size=[3, 3], padding='valid', activation=tf.nn.elu,
                             name='5CNN', kernel_initializer=tf.contrib.layers.variance_scaling_initializer())

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

features_definition = [conv1, conv2, conv3, conv4, conv5]

if not config['load_extracted_features']:  # extract and store features
    features_path = str(config['model_name']) + '_' + str(config['pre_processing']['n_frames']) + \
                    '_' + str(str(config['pre_processing']['n_mel_bands'])) + str(int(time.time()))

    # extract features for train-set
    x_train, y_train = format_data(prefix='data/GTZAN/audio',
                                         list_audios='data/GTZAN/idx/train_dev.txt')

    # extract features for val-set
    x_val, y_val = format_data(prefix='data/GTZAN/audio',
                                     list_audios='data/GTZAN/idx/val_dev.txt')

    # extract features for test-set
    x_test, y_test = format_data(prefix='data/GTZAN/audio',
                                       list_audios='data/GTZAN/idx/test_dev.txt')

    # storing extacted features
    with open('data/GTZAN/features/' + features_path + '.pkl', 'wb') as f:
        pickle.dump([x_train, y_train, x_val, y_val, x_test, y_test], f)

else:  # load extracted features
    print('Loading features: ' + config['load_extracted_features'])
    with open(config['load_extracted_features'], 'rb') as f:
        x_train, y_train, x_val, y_val, x_test, y_test = pickle.load(f)

# Train SVM
clf = SVC()
clf.fit(x_train, y_train)
SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False)
# Validate hyper-parameters
#
#
# # Evaluate
#
print("val-0: ")
print(clf.predict([x_val[0]]))
print(y_val[0])

print("val-1: ")
print(clf.predict([x_val[1]]))
print(y_val[1])

print("val-2: ")
print(clf.predict([x_val[2]]))
print(y_val[2])

print("val-3: ")
print(clf.predict([x_val[3]]))
print(y_val[3])

# Paralelitzar carregar audios
# hyper-parameter search
