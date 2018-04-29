import numpy as np
import tensorflow as tf
import librosa
import pickle
import time
import os
import datasets
import dl_models
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.cross_validation import PredefinedSplit
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.svm import LinearSVC
from elm import GenELMClassifier
from random_layer import RandomLayer
from sklearn import linear_model

from config_file import config_main
config = config_main

svm_params = [

    {'kernel': ['rbf'],
     'gamma': [1 / (2 ** 3), 1 / (2 ** 5), 1 / (2 ** 7), 1 / (2 ** 9), 1 / (2 ** 11), 1 / (2 ** 13), 'auto'],
     'C': [0.1, 2.0, 8.0, 32.0]},

    {'kernel': ['linear'],
     'C': [0.1, 2.0, 8.0, 32.0]}

]

if config['model_type'] == 'linearSVM':
    hyperparameters = [0.1, 1.0, 2.0]
elif config['model_type'] == 'ELM':
    hyperparameters = [100, 250, 500, 1200, 1800, 2500] # remove 100 and 250
elif config['model_type'] == 'KNN':
    hyperparameters = [1,3,5,10,20,30,50,100]

#------#
# CNNs #
#------#

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
                data = compute_input(file_path,config['sampling_rate'])
                first = False
            else:
                data = np.append(data,compute_input(file_path,config['sampling_rate']), axis=0)
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
                data = compute_input(file_path,config['sampling_rate'])
                first = False
            else:
                data = np.append(data,compute_input(file_path,config['sampling_rate']), axis=0)
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
                    if len(feature_maps[i].shape) == 3: # compute mean 1D-feature map
                        tmp_features[i, m] = np.mean(np.squeeze(feature_maps[i][j, :, m]))
                    elif len(feature_maps[i].shape) == 4: # compute mean 2D-feature map
                        tmp_features[i, m] = np.mean(np.squeeze(feature_maps[i][j, :, :, m]))
            X.append(tmp_features)
            Y.append(batch[1][j]) 
            ID.append(batch[2][j])
        print('Shape Annotations: ' + str(np.array(Y).shape))
        print('Shape X: ' + str(np.array(X).shape))
        print('# IDs: ' + str(np.array(ID).shape[0]))
    return X, Y, ID


def compute_input(audio_path, sampling_rate):
    # compute spectrogram
    audio, sr = librosa.load(audio_path, sr=sampling_rate)

    if config['CNN']['signal'] == 'spectrogram':
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
        if len(src) < config['CNN']['n_frames']:
            if config['fix_length_by'] == 'zero-pad':
                print('Zero padding!')
                src_zeros = np.zeros((config['CNN']['n_frames'],config['CNN']['n_mels']))
                src_zeros[:len(src)] = src
                src = src_zeros

            elif config['fix_length_by'] == 'repeat-pad':
                print('Repeat and crop to the fixed_length!')
                src_repeat = src
                while (src_repeat.shape[0] < config['CNN']['n_frames']):
                    src_repeat = np.concatenate((src_repeat, src), axis=0)    
                src = src_repeat
                src = src[:config['CNN']['n_frames'], :]
        else:
            print('Cropping audio!')
            src = src[:config['CNN']['n_frames'], :]
    
    elif config['CNN']['signal'] == 'waveform':

        # zero-pad, repeat-pad and corpping are different in CNNs for having fixed-lengths patches in CNNs
        if len(audio) < config['CNN']['n_samples']:
            if config['fix_length_by'] == 'zero-pad':
                print('Zero padding!')
                src_zeros = np.zeros(config['CNN']['n_samples'])
                src_zeros[:len(audio)] = audio
                src = src_zeros

            elif config['fix_length_by'] == 'repeat-pad':
                print('Repeat and crop to the fixed_length!')
                src_repeat = audio
                while (len(src_repeat) < config['CNN']['n_samples']):
                    src_repeat = np.concatenate((src_repeat, audio), axis=0)    
                src = src_repeat
                src = src[:config['CNN']['n_samples']]
        else:
            print('Cropping audio!')
            src = audio[:config['CNN']['n_samples']]

        src = np.expand_dims(src, axis=1)  # let the matrix be
    
    audio_rep = np.expand_dims(src, axis=0) # let the tensor be
    return audio_rep


def select_cnn_feature_layers(feature_maps, selected_features_list):
    selected_features = []
    for i in range(len(feature_maps)):
        tmp = np.array([])
        for j in selected_features_list:
            tmp = np.concatenate((tmp, np.squeeze(feature_maps[i][j])))
        selected_features.append(tmp)
    return selected_features


#-------#
# MFCCs #
#-------#

def extract_mfcc_features(audio, sampling_rate=12000): 
    src, sr = librosa.load(audio, sr=sampling_rate)

    # zero-pad, repeat-pad and corpping are different in CNNs for having fixed-lengths patches in CNNs
    if config['fix_length_by'] == 'zero-pad' and len(src) < config['MFCC']['fixed_length']:
        print('Zero padding!')
        src_zeros = np.zeros(config['MFCC']['fixed_length']) # min length to have 3-frame mfcc's
        src_zeros[:len(src)] = src
        src = src_zeros
    elif config['fix_length_by'] == 'repeat-pad' and len(src) < config['MFCC']['fixed_length']:
        print('Repeat padding!')
        src_repeat = src
        while (len(src_repeat) < config['MFCC']['fixed_length']):
            src_repeat = np.concatenate((src_repeat, src), axis=0)    
        src = src_repeat
    elif config['fix_length_by'] == 'crop':
        print('Cropping audio!')
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
        print(str(n_song) + ': ' + song[:-1])
        X.append(extract_mfcc_features(prefix + song[:-1], config['sampling_rate']))
        Y.append(datasets.path2gt(song[:-1], config['dataset']))
        ID.append(song[:-1])
        n_song += 1
        print(Y)
        print(np.array(X).shape)
    return X, Y, ID


#-----------------------#
# CLASSIFICATION MODELS #
#-----------------------#

def define_classification_model(h):
    if config['model_type'] == 'linearSVM':
        return LinearSVC(C=h)
    elif config['model_type'] == 'ELM':
        rl = RandomLayer(n_hidden=h,activation_func='reclinear',alpha=1)
        return GenELMClassifier(hidden_layer = rl)
    elif config['model_type'] == 'MLP':
        return MLPClassifier(hidden_layer_sizes=(20,), max_iter=600, verbose=10, early_stopping=False)
    elif config['model_type'] == 'linear':
        return linear_model.SGDClassifier()
    elif config['model_type'] == 'KNN':
        return KNeighborsClassifier(n_neighbors=h)


if __name__ == '__main__':

    #--------------------#
    # FEATURE EXTRACTION #
    #--------------------#
    
    print(config)

    print('Set file name (unique identifier) for the experiment..')
    if config['features_type'] == 'MFCC':
        experiment_name = str(config['experiment_name']) + '_MFCC_' + str(int(time.time()))
    elif config['features_type'] == 'CNN':
        experiment_name = str(config['experiment_name']) + '_CNN_' \
            + '_' + str(config['model_type']) \
            + '_' + str(config['CNN']['signal']) \
            + '_' + str(config['CNN']['architecture']) \
            + '_' + str(config['CNN']['selected_features_list']) + '_'+ str(int(time.time()))
    print(experiment_name)

    if not config['load_extracted_features']: # extract features: MFCC of CNN

        print('Extracting features..')
        if config['features_type'] == 'CNN':
            if config['CNN']['signal'] == 'spectrogram':
                x_in = tf.placeholder(tf.float32, [None, None, config['CNN']['n_mels']])
            elif config['CNN']['signal'] == 'waveform':
                x_in = tf.placeholder(tf.float32, [None, config['CNN']['n_samples'], 1])             
            features_definition = dl_models.build(config, x_in)
            sess = tf.InteractiveSession()
            sess.run(tf.global_variables_initializer())
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
    f = open(config['results_folder'] + experiment_name + '.txt','w')

    if config['audios_list'] == False:
        print('train/val/test partitions are pre-defined!')
        if config['model_type'] == 'SVM':
            # hyperparameter search in val set
            x_dev = np.concatenate((x_train, x_val), axis=0)
            y_dev = np.concatenate((y_train, y_val), axis=0)
            val_mask = np.concatenate((-np.ones(len(y_train)), np.zeros(len(y_val))), axis=0)
            ps = PredefinedSplit(test_fold=val_mask)
            svc = SVC()
            hps = GridSearchCV(svc, svm_params, cv=ps, n_jobs=3, pre_dispatch=3*8, verbose=config['SVM_verbose']).fit(x_dev, y_dev)
            print('Best hyperparameter: ' + str(hps.best_params_))
            # define final model
            model = SVC()
            model.set_params(**hps.best_params_)
        else:
            score_max = 0
            h_max = -1
            for h in hyperparameters: # select best params in validation set
                print('Now in: ' + str(h))
                model = define_classification_model(h)
                model.fit(x_train, y_train) 
                score = accuracy_score(y_val, model.predict(x_val))
                print('- Score: ' + str(score))
                if score > score_max:
                    score_max = score
                    h_max = h
                print('Accuracy val set: ' + str(score_max))
                print('Best hyperparameter: ' + str(h_max))
                f.write('Accuracy val set: ' + str(score_max) + '\n')
                f.write('Best hyperparameter: ' + str(h_max))
                model = define_classification_model(h_max)

        # train model with best hyperparameters and evaluate in test set
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        print('Detailed classification report: ')
        print(classification_report(y_test, y_pred))
        print('Accuracy test set: ')
        print(accuracy_score(y_test, y_pred))

        print(config)

        print('Storing results..')   
        f.write(str(classification_report(y_test, y_pred)))     
        f.write('Accuracy test set: ' + str(accuracy_score(y_test, y_pred)) + '\n')
        f.write(str(config))

    else:
        print('10 fold cross-validation!')

        if config['dataset'] == 'UrbanSound8K':
            print('UrbanSound8K dataset with pre-defined splits!')
            df = pd.read_csv('/datasets/MTG/users/jpons/urban_sounds/UrbanSound8K/metadata/UrbanSound8K.csv')
            folds_mask = []
            for i in ids:
                tag = i[i.rfind('/')+1:]
                folds_mask.append(int(df[df.slice_file_name==tag].fold))
            ps = PredefinedSplit(test_fold=folds_mask)

        else:
            ps = 10

        if config['model_type'] == 'SVM':
            svc = SVC()
            model = GridSearchCV(svc, svm_params, cv=ps, n_jobs=3, pre_dispatch=3*8, verbose=config['SVM_verbose']).fit(x, y)
            print('[SVM] Best score of ' + str(model.best_score_) + ': ' + str(model.best_params_))
            f.write('[SVM] Best score of ' + str(model.best_score_) + ': ' + str(model.best_params_))

        else:
            score_max = 0
            h_max = -1
            for h in hyperparameters:
                print('Now in: ' + str(h))
                model = define_classification_model(h)
                scores = cross_val_score(model, x, y, cv=ps, scoring='accuracy')
                print('- Score: ' + str(scores.mean()))
                if scores.mean() > score_max:
                    h_max = h
                    score_max = scores.mean()
            print(config['model_type'] + ' - score of best model: ' + str(score_max) + ' with ' + str(h_max))
            f.write(config['model_type'] + ' - score of best model: ' + str(score_max) + ' with ' + str(h_max))

        print(config)
        f.write(str(config))

    f.close()

# NOTES ON SPECTROGRAM. Mel power spectrogram. Sampling rate: 12k. fmin=0 and fmax=6000. Using shorter clips.

# IDEAS: - Check statistics of input data (zero-mean/one-var)?
#        - Only store mean values for features?
