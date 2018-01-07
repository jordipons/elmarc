import os
import sys
import numpy as np
import keras
from argparse import Namespace
import pandas as pd
import librosa
import time
from multiprocessing import Pool
from joblib import Parallel, delayed
from keras import backend as K
import KC_model

config = {
    'experiment_name': 'Choi_new_code_istrainFalse_woDrop',
    'pre_processing': {
        'n_mels': 96,
        'n_frames': 1360  # number of time-frames of hte input time-freq representation
    },
    'features_type': 'CNN',
    'selected_features_list': [0, 1, 2, 3, 4],
    'load_extracted_features': False,
    # load already extracted features by defining the
    # path where these are saved - set as False for extracting those again
    'audio_path': '/mnt/vmdata/users/jpons/GTZAN/',
    'audios_list': '/mnt/vmdata/users/jpons/GTZAN_no_partitions/list.txt',
    'save_extracted_features_folder': '../data/GTZAN/features/',
    'sr': 12000,
    'len_src': 29, # seconds
    'n_jobs': 4, # or 9?
    'ref_n_src' = 12000 * 29
}


def gen_audiofiles(filenames, batch_size=256):
    '''gen single audio file src in a batch_size=1 form for keras model.predict_generator
    df: dataframe 
    total_size: integer.
    batch_size: integer.
    dataroot: root path for data'''

    ''''''
    pool = Pool(N_JOBS)
    def _multi_loading(pool, paths):
        srcs = pool.map(_load_audio, paths)
        srcs = np.array(srcs)
        try:
            srcs = srcs[:, np.newaxis, :]
        except:
            pdb.set_trace()

        return srcs
    
    total_size = len(filenames) # CHECK THIS!
    n_leftover = int(total_size % batch_size)
    leftover = n_leftover != 0
    n_batch = int(total_size / batch_size)
    print('n_batch: {}, n_leftover: {}, all: {}'.format(n_batch, n_leftover, total_size))
    
    for batch_idx in xrange(n_batch):
        paths = []
        for inbatch_idx in range(batch_size):
            paths.append(gen_f.next()) # TODO: append file path!
        print('..yielding {}/{} batch..'.format(batch_idx, n_batch))                    
        yield _multi_loading(pool, paths)
        
    if leftover:
        paths = []
        for inbatch_idx in range(n_leftover):
            paths.append(gen_f.next())
        print('..yielding final batch w {} data sample..'.format(len(paths)))
        yield _multi_loading(pool, paths)



def _load_audio(path):
    """Load audio file at path with sampling rate=SR, duration=len_src, and return it"""
    src, sr = librosa.load(path, sr=SR, duration=len_src * SR / float(SR))
    src = src[:config['ref_n_src']]
    result = np.zeros(config['ref_n_src'])
    result[:len(src)] = src[:config['ref_n_src']]

    audio_rep = librosa.feature.melspectrogram(y=result,
                                               sr=sr,
                                               hop_length=256,
                                               n_fft=512,
                                               n_mels=config['pre_processing']['n_mels'],
                                               power=2,
                                               fmin=0.0,
                                               fmax=6000.0)

    # normalize audio representation
    # audio_rep = np.log10(10000 * audio_rep + 1)
    # audio_rep = (audio_rep - config['patches_params']['mean']) / config['patches_params']['std']
    audio_rep = librosa.core.logamplitude(audio_rep)
    #audio_rep = np.expand_dims(audio_rep, axis=0)
    #audio_rep = audio_rep[:, :config['pre_processing']['n_frames'], :]

    return audio_rep



def predict(filenames, batch_size, model):
    """Extract convnet feature using given model"""
    gen_audio = gen_audiofiles(filenames, batch_size)
    feats = model.predict_generator(generator=gen_audio, 
                                    val_samples=len(filenames), 
                                    max_q_size=1)



def load_model_for_mid(mid_idx):
    assert 0 <= mid_idx <= 4
    args = Namespace(test=False, data_percent=100, model_name='', tf_type='melgram',
                     normalize='no', decibel=True, fmin=0.0, fmax=6000,
                     n_mels=96, trainable_fb=False, trainable_kernel=False,
                     conv_until=mid_idx)
    model = build_convnet_model(args, last_layer=False)
    return model


if __name__ == '__main__':

    songs_list = open(DATASET_LIST, 'r')
    model = load_model_for_mid(1) # MORE LAYERS!
    predict(songs_list, batch_size, model)


