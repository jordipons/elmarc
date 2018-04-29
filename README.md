## Randomly weighted CNNs for (music) audio classification
The computer vision literature shows that randomly weighted neural networks perform reasonably as feature extractors. Following this idea, we study how non-trained (randomly weighted) convolutional neural networks perform as feature extractors for (music) audio classification tasks. We use features extracted from the embeddings of
deep architectures as input to a classifier – with the goal to compare classification accuracies when using different randomly weighted architectures. By following this methodology, we run a comprehensive evaluation of the current deep architectures for audio classification.

This study builds on top of prior works showing that the (classification) performance delivered by random CNN
features is correlated with the results of their end-to-end trained counterparts [40, 42]. We use this property to run a comprehensive evaluation of current deep architectures for (music) audio. Our method is as follows: first, we
extract a feature vector from the embeddings of a randomly weighted CNN; and then, we input these features
to a classifier – which can be an SVM or an ELM. Our goal is to compare the obtained classification accuracies
when using different CNN architectures. We consider three datasets for our study; the first one, is the fault-filtered GTZAN dataset for music genre classification (classification accuracy is the figure-of-merit expressed in the vertical axis of the following figures):
<p align="center"><img src="img/GTZAN3500.png" height="290"></p>
The second dataset is the Extended Ballroom (with rhythm/tempo music classes):
<p align="center"><img src="img/Ball3500.png" height="290"></p>
And the last dataset is the Urban Sounds 8k, composed of natural (non-music) sounds:
<p align="center"><img src="img/us8k3500.png" height="290"></p>

## Observations
- **Extended Ballroom**: (musical) priors embedded in the structure of the model can facilitate capturing useful (temporal) cues for classifying rhythm/tempo classes – see the accuracy performance of the Temporal architecture (89.82%), which is very close to the state-of-the-art (94.9%).

- **All datasets**: the results we obtain are far from random, since: *(i)* randomly weighted CNNs are (in some cases) close to match the accuracies obtained by trained CNNs; and *(ii)* these are able to outperform MFCCs. 

- **Waveform front-ends**: sample-level >> frame-level many-shapes > frame-level – as noted in the (trained) literature [26, 51, 52]. 

- **Spectrogram front-ends**: 7x96<7x86 – as shown in prior (trained) works [30, 35]. 

## Usage
