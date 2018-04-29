## Randomly weighted CNNs for (music) audio classification
The computer vision literature shows that randomly weighted neural networks perform reasonably as feature extractors. Following this idea, we study how non-trained (randomly weighted) convolutional neural networks perform as feature extractors for (music) audio classification tasks. We use features extracted from the embeddings of
deep architectures as input to a classifier – with the goal to compare classification accuracies when using different randomly weighted architectures. By following this methodology, we run a comprehensive evaluation of the current deep architectures for audio classification.

This study builds on top of prior works showing that the (classification) performance delivered by random CNN features is correlated with the results of their end-to-end trained counterparts [1]. We use this property to run a comprehensive evaluation of the current deep architectures for (music) audio. Our method works as follows: first, we extract a feature vector from the embeddings of a randomly weighted CNN; and then, we input these features to a classifier – which can be an support vector machine (SVM) or an extreme learning machine (ELM). Our goal is to compare the obtained classification accuracies when using different CNN architectures. 

We consider three datasets for our study. The first one, is the fault-filtered GTZAN dataset for music genre classification – classification accuracy is the figure-of-merit expressed in the vertical axis of the following figures:

<p align="center"><img src="img/GTZAN3500.png" height="290"></p>
The second dataset is the Extended Ballroom (meant to discriminate rhythm/tempo music classes):
<p align="center"><img src="img/Ball3500.png" height="290"></p>
And the last dataset is the Urban Sounds 8k, composed of natural (non-music) sounds:
<p align="center"><img src="img/us8k3500.png" height="290"></p>

## Observations

- *All datasets*: the results we obtain are far from random, since: 1) randomly weighted CNNs are (in some cases) close to match the accuracies obtained by trained CNNs - see the paper; and 2) these are able to outperform MFCCs. 

- *Extended Ballroom dataset*: (musical) priors embedded in the structure of the model can facilitate capturing useful (temporal) cues for classifying rhythm/tempo classes – see the accuracy performance of the Temporal architecture (89.82%), which is very close to the state-of-the-art (94.9%).

- *Waveform front-ends*: sample-level >> frame-level many-shapes > frame-level – as noted in the (trained) literature. 

- *Spectrogram front-ends*: 7x96 < 7x86 – as shown in prior (trained) works. 

## Dependencies
You need to install the following dependencies: tensorflow, librosa, pandas, numpy, scipy, sklearn, pickle. It is not a bad idea to run these models on CPUs – therefore, we recommend to install the CPU version of tensorflow.

The public extreme learning machine implementation we use (already included in this repo) can be found here: https://github.com/zygmuntz/Python-ELM

## Usage

Set `src/config_file.py` and run: `python main.py`

Some documentation is available in `config_file.py`, but here an example of how to set the configuration file:

```python
config_main = {

    'experiment_name': 'v0_or_any_name',
    'features_type': 'CNN',
    'model_type': 'SVM',
    'SVM_verbose': 1,
    'load_extracted_features': False,
    'sampling_rate': 12000,

    'dataset': 'GTZAN',
    'audio_path': '/datasets/MTG/users/jpons/GTZAN/',
    'save_extracted_features_folder': '../data/GTZAN/features/',
    'results_folder': '../data/GTZAN/results/',
    'train_set_list': '/datasets/MTG/users/jpons/GTZAN_partitions/train_filtered.txt',
    'val_set_list': '/datasets/MTG/users/jpons/GTZAN_partitions/valid_filtered.txt',
    'test_set_list': '/datasets/MTG/users/jpons/GTZAN_partitions/test_filtered.txt',
    'audios_list': False,
    
    'CNN': {
        'batch_size': 5,

        'signal': 'waveform',
        'n_samples': 350000,

        'architecture': 'sample_level',
        'num_filters': 512,
        'selected_features_list': [0, 1, 2, 3, 4, 5, 6],
        
    }
}
```
This experiment runs the `sample_level` CNN architecture with 512 filters in every layer, and we use every feature map in every layer to compute the feature vector – see the implementation of the `sample_level` model at `src/dl_models.py`.

The input waveforms of the GTZAN dataset are formatted to be of approx. 29sec (350,000 samples at 12kHz), features are computed in batches of 5, and we use an SVM as classifier.


## References
[1] Saxe, et al. On Random Weights and Unsupervised Feature Learning. In: ICML. 2011. p. 1089-1096.
