import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import librosa
import pickle
import time
import os
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.cross_validation import PredefinedSplit

config = {
    'experiment_name': 'MFCC_choi_all_GTZAN_audios',
    'features_type': 'MFCC',
    'load_extracted_features': '/mnt/vmdata/users/jpons/elmarc/data/GTZAN/features/MFCC_choi_all_GTZAN_audios_1514464039.pkl',
    # load already extracted features by defining the
    # path where these are saved - set as False for extracting those again
    'audio_path': '/mnt/vmdata/users/jpons/GTZAN/',
    'audios_list': '/mnt/vmdata/users/jpons/GTZAN_no_partitions/list.txt',
    'save_extracted_features_folder': '../data/GTZAN/features/'
}

#svm_params = [
#    {'kernel': ['rbf'],
#     'gamma': [1 / (2 ** 3), 1 / (2 ** 5), 1 / (2 ** 7), 1 / (2 ** 9), 1 / (2 ** 11), 'auto'],
#     'C': [0.1, 2, 8, 32, 100, 1000]}
#]

svm_params = [
    {'kernel': ['rbf'],
     'gamma': [1 / (2 ** 3), 1 / (2 ** 5), 1 / (2 ** 7), 1 / (2 ** 9), 1 / (2 ** 11), 1 / (2 ** 13), 'auto'],
     'C': [0.1, 2.0, 8.0, 32.0]},
    {'kernel': ['linear'],
     'C': [0.1, 2.0, 8.0, 32.0]}
]

def extract_features(audio, feature_type, sampling_rate=16000):
    """
    Extract tensor-flow features: extract audio, compute librosa features and
    pass it through the tensor-flow model to extract the *features_list*

    :param audio: String pointing where the audio is located
    :param sampling_rate: Sampling rate used when loading the audio (change it for down-sampling)

    :return features: Extracted features per *audio* song
    """
    if feature_type == 'MFCC':

        src_zeros = np.zeros(1024) # min length to have 3-frame mfcc's
        src, sr = librosa.load(audio, sr=sampling_rate, duration=29.) # max len: 29s, can be shorter.
        if len(src) < 1024:
            src_zeros[:len(src)] = src
            src = src_zeros
    
        mfcc = librosa.feature.mfcc(src, sampling_rate, n_mfcc=20)
        dmfcc = mfcc[:, 1:] - mfcc[:, :-1]
        ddmfcc = dmfcc[:, 1:] - dmfcc[:, :-1]
        return np.concatenate((np.mean(mfcc, axis=1), np.std(mfcc, axis=1),
                               np.mean(dmfcc, axis=1), np.std(dmfcc, axis=1),
                               np.mean(ddmfcc, axis=1), np.std(ddmfcc, axis=1)), 
                               axis=0)

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
        print(ground_truth)
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


if __name__ == '__main__':

    #--------------------#
    # FEATURE EXTRACTION #
    #--------------------#

    if not config['load_extracted_features']:  # extract and store features

        features_path = str(config['experiment_name']) + '_' + str(int(time.time()))

        print('Extract features..')
        x, y = format_data_gtzan(prefix=config['audio_path'],
                                                 list_audios=config['audios_list'],
                                                 features_type=config['features_type'])


        # storing extacted features
        if not os.path.exists(config['save_extracted_features_folder']):
            os.makedirs(config['save_extracted_features_folder'])
        with open(config['save_extracted_features_folder'] + features_path + '.pkl', 'wb') as f:
            pickle.dump([x, y], f)

    else:  # load extracted features
        
        print('Loading features: ' + config['load_extracted_features'])

        with open(config['load_extracted_features'], 'rb') as f:
            x, y = pickle.load(f)


    #------------#
    # CLASSIFIER #
    #------------#

    svc = SVC()
    svm_hps = GridSearchCV(svc, svm_params, cv=10, n_jobs=3, pre_dispatch=3*8).fit(x, y)
    print('best score of {}: {}'.format(svm_hps.best_score_,svm_hps.best_params_))
    print(svm_hps.best_score_)

# COMMENT 1: with linear kernel as option, results get much worse in test-set.
# BEST RESULT MFCC: 0.524, with rbf kernel only.
# BEST RESULT BN-CNN: 0.431, with rbf kernel only.
