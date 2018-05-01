## Randomly weighted CNNs for (music) audio classification
The computer vision literature shows that randomly weighted neural networks perform reasonably as feature extractors. Following this idea, we study how non-trained (randomly weighted) convolutional neural networks perform as feature extractors for (music) audio classification tasks. We use features extracted from the embeddings of deep architectures as input to a classifier – with the goal to compare classification accuracies when using different randomly weighted architectures. By following this methodology, we run a fast and comprehensive evaluation of the current deep architectures for audio classification.

This study builds on top of prior works showing that the (classification) performance delivered by random CNN features is correlated with the results of their end-to-end trained counterparts [1]. Put in other words: architectures which perform well with random weights also tend to perform well with trained weights. We use this property to run a fast and comprehensive evaluation of the current deep architectures for (music) audio. Our method works as follows: first, we extract a feature vector from the embeddings of a randomly weighted CNN; and then, we input these features to a classifier – which can be a support vector machine (SVM) or an extreme learning machine (ELM). Our goal is to compare the obtained classification accuracies when using different CNN architectures. 

We consider three datasets in our study. The first one, is the fault-filtered GTZAN dataset for music genre classification – classification accuracy is the figure-of-merit expressed in the vertical axis of the following figures:

<p align="center"><img src="img/GTZAN3500.png" height="290"></p>
The second dataset is the Extended Ballroom (meant to learn how to discriminate rhythm/tempo music classes):
<p align="center"><img src="img/Ball3500.png" height="290"></p>
And the last dataset is the Urban Sounds 8k, composed of (non-music) sounds:
<p align="center"><img src="img/us8k3500.png" height="290"></p>

## Observations

- The results we obtain are far from random, since: *1)* randomly weighted CNNs are (in some cases) close to match the accuracies obtained by trained CNNs – see the paper; and *2)* these are able to outperform MFCCs. 

- *Extended Ballroom dataset experiments*: (musical) priors embedded in the structure of the model facilitate capturing useful (temporal) cues for classifying rhythm/tempo classes – see the accuracy performance of the (non-trained) Temporal architecture (89.82%), which is very close to the state-of-the-art (93.7%).

- *Waveform front-ends*: sample-level >> frame-level many-shapes > frame-level – as noted in the (trained) literature. 

- *Spectrogram front-ends*: 7x96 < 7x86 – as shown in prior (trained) works. 

## Dependencies
You need to install the following dependencies: tensorflow, librosa, pandas, numpy, scipy, sklearn, pickle. It is not a bad idea to run these models on a CPU – therefore, we recommend to install the CPU version of tensorflow.

The public extreme learning machine implementation we use (already included in this repo) can be found [here](https://github.com/zygmuntz/Python-ELM).

## Usage

Set `src/config_file.py` and run: `python main.py`

Some documentation is available in `config_file.py`, but here some examples in how to set the configuration file:

```python
config_main = {

    # Experimental setup
    'experiment_name': 'v0_or_any_name',
    'features_type': 'CNN',
    'model_type': 'SVM',
    'SVM_verbose': 1,
    'load_extracted_features': False,
    'sampling_rate': 12000,

    # Dataset configuration
    'dataset': 'GTZAN',
    'audio_path': '/path_to_audio/jpons/GTZAN/',
    'save_extracted_features_folder': '../data/GTZAN/features/',
    'results_folder': '../data/GTZAN/results/',
    'train_set_list': '/path_to_train_set/jpons/GTZAN_partitions/train_filtered.txt',
    'val_set_list': '/path_to_val_set/jpons/GTZAN_partitions/valid_filtered.txt',
    'test_set_list': '/path_to_test_set/jpons/GTZAN_partitions/test_filtered.txt',
    'audios_list': False,

    # Waveform model: sample level CNN
    'CNN': {
        'signal': 'waveform',
        'n_samples': 350000,

        'architecture': 'sample_level',
        'num_filters': 512,
        'selected_features_list': [0, 1, 2, 3, 4, 5, 6],

        'batch_size': 5
    }
}
```
As a result of this config file: the input waveforms of the GTZAN dataset are formatted to be of `≈` 29sec (350,000 samples at 12kHz), features are computed in batches of 5, and we use an SVM classifier.

This experiment runs the `sample_level` CNN architecture with 512 filters in every layer, and we use every feature map in every layer to compute the feature vector – see the implementation of the `sample_level` model at `src/dl_models.py`.

```python
config_main = {

    # Experimental setup
    'experiment_name': 'v0_or_any_name',
    'features_type': 'CNN',
    'model_type': 'ELM',
    'load_extracted_features': False,
    'sampling_rate': 12000,

    # Dataset configuration
    'dataset': 'ExtendedBallroom',
    'audio_path': '/path_to_audio/jpons/extended_ballroom/',
    'save_extracted_features_folder': '../data/Extended_Ballroom/features/',
    'results_folder': '../data/Extended_Ballroom/results/',
    'train_set_list': None,
    'val_set_list': None,
    'test_set_list': None,
    'audios_list': '/path_to_all_audios/jpons/extended_ballroom/all_files.txt',

    # Spectrogram model: 7x86 CNN
    'CNN': {
        'signal': 'spectrogram',
        'n_mels': 96,
        'n_frames': 1376,

        'architecture': 'cnn_single',
        'num_filters': 3585,
        'selected_features_list': [1],
        'filter_shape': [7,86],
        'pool_shape': [1,11],

        'batch_size': 5        
    }
}
```
As a result of this config file: the input spectrograms of the Extended Ballroom dataset are formatted to be of `≈` 29sec (1376 frames at 12kHz), features are computed in batches of 5, and we use an ELM classifier.

This experiment runs the `7x86` CNN architecture with 3585 filters, and we use every feature map (of this single-layered CNN) to compute the feature vector – see the implementation of the `7x86` model at `src/dl_models.py`.

## Reproducing our results

To reproduce our results, one needs to download the data and use the same partitions:

- **GTZAN fault-filtered version**: download the data [(link)](http://marsyasweb.appspot.com/download/data_sets/). Download the (.txt) files that list which audios are in every partition [(link)](https://github.com/jongpillee/music_dataset_split/tree/master/GTZAN_split). Set the config file: `'train_set_list': 'path/train_audios.txt'`, `'val_set_list': 'path/val_audios.txt'`, `'test_set_list': 'path/test_audios.txt'`, `'audios_list': False`.

- **Extended Ballroom**: download the data [(link)](http://anasynth.ircam.fr/home/media/ExtendedBallroom). 10 stratified folds will be randomly generated (via a sklearn function) for cross-validation. List all the audios in a file, and set the config file: `'train_set_list': None`, `'val_set_list': None`, `'test_set_list': None`, `'audios_list': 'path/all_audios.txt'`.

- **Urban Sound 8k**: download the data [(link)](https://serv.cusp.nyu.edu/projects/urbansounddataset/urbansound8k.html). Partitions were already defined by the dataset authors, and we have some code to get those! Just list all the audios in a file, and set the config file: `'train_set_list': None`, `'val_set_list': None`, `'test_set_list': None`, `'audios_list': 'path/all_audios.txt'`.

For more information, see the documentation available in `config_file.py`.

## References
[1] Saxe, et al. On Random Weights and Unsupervised Feature Learning. In: ICML. 2011. p. 1089-1096.
