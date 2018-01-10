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

    'experiment_name': 'v0',
    'features_type': 'CNN', # CNN or MFCC

    'load_extracted_features': False,
    'audio_path': '/mnt/vmdata/users/jpons/GTZAN/',
    'audios_list': '/mnt/vmdata/users/jpons/GTZAN_no_partitions_random/list_random.txt',
    'save_extracted_features_folder': '../data/GTZAN/features/', 
    'results_folder': '../data/GTZAN/results/',
   
    'sampling_rate': 12000,

    'CNN': {
        'n_mels': 96,
        'n_frames': 1404, # GTZAN: 1407
        'batch_size': 10,
        'is_train': False, 

        'architecture': 'cnn_small_filters',
        'num_filters': 32, # 90 or 32
        'selected_features_list': [0, 1, 2, 3, 4]

        #'architecture': 'cnn_music',
        #'num_filters': 16, # 128, 64, 32, 16, 8 or 4
        #'selected_features_list': [0] # timbral [0], temporal [1] or both [0, 1]
    },
}


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
        for i in range(start_i,start_i + batchsize,1):
            file_path = prefix + audio_paths_list[i]
            file_path = file_path[:-1] # remove /n
            tag = audio_paths_list[i][:audio_paths_list[i].rfind('/')]
            print(str(i) + ': ' + file_path)
            if first:
                data = compute_spectrogram(file_path,config['sampling_rate'])
                first = False
            else:
                data = np.append(data,compute_spectrogram(file_path,config['sampling_rate']), axis=0)
            ground_truth.append(gtzan_ground_truth(tag))
        yield data, ground_truth

    if leftover:
        first = True
        ground_truth = []
        for i in range(start_i + batchsize, start_i + batchsize + n_leftover,1):
            file_path = prefix + audio_paths_list[i]
            file_path = file_path[:-1] # remove /n
            tag = audio_paths_list[i][:audio_paths_list[i].rfind('/')]
            print(str(i) + ': ' + file_path)
            if first:
                data = compute_spectrogram(file_path,config['sampling_rate'])
                first = False
            else:
                data = np.append(data,compute_spectrogram(file_path,config['sampling_rate']), axis=0)
            ground_truth.append(gtzan_ground_truth(tag))
        yield data, ground_truth


def format_cnn_data(prefix, list_audios):
    l_audios = open(list_audios, 'r')
    audio_paths_list = []
    for s in l_audios:
        audio_paths_list.append(s)
    X = []
    Y = []
    for batch in iterate_minibatches(prefix, audio_paths_list, config['CNN']['batch_size']):      
        # feature_maps[i][j, k, l, m]
        # i: layer where extracted the feature
        # j: batch-sample dimension
        # k: one feature-map axis
        # l: other feature-map axis
        # m: feature-map
        feature_maps = sess.run(features_definition, feed_dict={x: batch[0], is_train: config['CNN']['is_train']})
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
        print(Y)
        print(np.array(X).shape)
    
    return X, Y


def compute_spectrogram(audio_path, sampling_rate):
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
    audio_rep = librosa.core.logamplitude(audio_rep)
    audio_rep = np.expand_dims(audio_rep, axis=0)
    audio_rep = audio_rep[:, :config['CNN']['n_frames'], :] # cropping signal to n_frames
    return audio_rep

def cnn_small_filters():

    with tf.name_scope('cnn_as_choi'):
        global x
        x = tf.placeholder(tf.float32, [None, None, config['CNN']['n_mels']])

        global is_train
        is_train = tf.placeholder(tf.bool)

        print('Input: ' + str(x.get_shape))

        input_layer = tf.reshape(x,[-1, config['CNN']['n_frames'], config['CNN']['n_mels'], 1])
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
    print('BU!')

    return [pool1, pool2, pool3, pool4, pool5]


def cnn_music():
   
    # remove some temporal filters to have the same ammount of timbral and temporal filters
    if config['CNN']['num_filters'] == 128:
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
        global x
        x = tf.placeholder(tf.float32, [None, None, config['CNN']['n_mels']])

        global is_train
        is_train = tf.placeholder(tf.bool)

        print('Input: ' + str(x.get_shape))

        input_layer = tf.reshape(x, [-1, config['CNN']['n_frames'], config['CNN']['n_mels'], 1])

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


# GTZAN

def gtzan_ground_truth(ground_truth):

    if ground_truth == 'blues':
        return 0
    elif ground_truth == 'classical':
        return 1
    elif ground_truth == 'country':
        return 2
    elif ground_truth == 'disco':
        return 3
    elif ground_truth == 'hiphop':
        return 4
    elif ground_truth == 'jazz':
        return 5
    elif ground_truth == 'metal':
        return 6
    elif ground_truth == 'pop':
        return 7
    elif ground_truth == 'reggae':
        return 8
    elif ground_truth == 'rock':
        return 9
    else:
        print('Warning: did not find the corresponding ground truth (' + str(ground_truth) + ').')
        import ipdb; ipdb.set_trace()


# MFCCs

def extract_mfcc_features(audio, sampling_rate=12000): 
    src, sr = librosa.load(audio, sr=sampling_rate)
    
    # dmfcc as in https://github.com/keunwoochoi/transfer_learning_music/
    mfcc = librosa.feature.mfcc(src, sampling_rate, n_mfcc=20)
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
    n_song = 0
    for song in songs_list:
        ground_truth = song[:song.rfind('/')]
        print(str(n_song) + ': ' + song[:-1])
        X.append(extract_mfcc_features(prefix + song[:-1], config['sampling_rate']))
        Y.append(gtzan_ground_truth(ground_truth))
        n_song += 1
        print(Y)
        print(np.array(X).shape)
    return X, Y


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
                + '_' + str(config['CNN']['is_train']) + '_' + str(config['CNN']['architecture']) \
                + '_' + str(config['CNN']['num_filters']) + '_' + str(config['CNN']['selected_features_list']) \
                + '_'+ str(int(time.time()))

        print(experiment_name)

        print('Extracting features..')

        if config['features_type'] == 'CNN':

            if config['CNN']['architecture'] == 'cnn_small_filters':
                features_definition = cnn_small_filters()
            elif config['CNN']['architecture'] == 'cnn_music':
                features_definition = cnn_music()

            print('Number of parameters of the model: ' + str(
                   count_params(tf.trainable_variables()))+'\n')

            x, y = format_cnn_data(prefix=config['audio_path'],
                                    list_audios=config['audios_list'])

        elif config['features_type'] == 'MFCC':

            x, y = format_mfcc_data(prefix=config['audio_path'],
                                    list_audios=config['audios_list'])


        print('Storing extracted features..')        

        if not os.path.exists(config['save_extracted_features_folder']):
            os.makedirs(config['save_extracted_features_folder'])

        with open(config['save_extracted_features_folder'] + experiment_name + '.pkl', 'wb') as f:
            pickle.dump([x, y, config], f)

    else:  # load extracted features
        
        print('Loading features: ' + config['load_extracted_features'])

        with open(config['load_extracted_features'], 'rb') as f:
            x, y, config = pickle.load(f)


    if config['features_type'] == 'CNN':

        print('Select CNN features..')

        x = select_cnn_feature_layers(x, config['CNN']['selected_features_list'])


    print('Data size (data points, feature vector)..')
    print(np.array(x).shape)

    #------------#
    # CLASSIFIER #
    #------------#

    svc = SVC()
    svm_hps = GridSearchCV(svc, svm_params, cv=10, n_jobs=3, pre_dispatch=3*8).fit(x, y)

    print('Storing extracted features..')        

    if not os.path.exists(config['results_folder']):
        os.makedirs(config['results_folder'])

    f = open(config['results_folder'] + experiment_name + '.txt','w')
    f.write('Best score of' + str(svm_hps.best_score_) + ':' + str(svm_hps.best_params_))
    f.write(str(config))
    f.close()

    print('Best score of {}: {}'.format(svm_hps.best_score_,svm_hps.best_params_))
    print(svm_hps.best_score_)
    print(config)

# NOTES ON SPECTROGRAM. Mel power spectrogram. Sampling rate: 12k. fmin=0 and fmax=6000. Using shorter clips.

# IDEAS. Check statistics of input data (zero-mean/one-var)?
