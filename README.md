# Contrastive Learning of Object Representations


<img align="right" width="300" height="" src="https://upload.wikimedia.org/wikipedia/commons/1/1e/Logo-Goethe-University-Frankfurt-am-Main.svg">


**Supervisor:**
* [Prof. Dr. Gemma Roig](http://www.cvai.cs.uni-frankfurt.de/team.html)


**Institutions:**
  * **[Goethe University](http://www.cvai.cs.uni-frankfurt.de/index.html)**
  * **[CVAI - Computational Vision & Artificial Intelligence](http://www.cvai.cs.uni-frankfurt.de/team.html)**
  


## Project Description

**Contrastive Learning** is an unsupervised method for learning similarities or differences in a dataset, whithout the need of labels. The main idea is to provide the machine with similar (so called positive samples) and with very different data (negative or corrupted samples). The task of the machine then is to leverage this information and to pull the positive examples in the embedded space together, while pushing the negative examples further apart. Next to being unsupervised, another major advantage is that the loss is applied on the latent space rather than being pixel-base. This saves computation and memory, because there is no need for a decoder and also delivers more accurate results.

In this work, we will investigate the novel **SetCon** model from **'Learning Object-Centric Video Models by Contrasting Sets' by Löwe et al.** [[1]](#1) ([Paper](https://arxiv.org/abs/2011.10287))
The SetCon model has been published in November 2020 by the Google Brain Team and introduces an attention-based object extraction in combination with contrastive learning. It incorporates a novel <em> slot-attention module </em> [[3]](#3)([Paper](https://arxiv.org/abs/2006.15055)), which is an iterative attention mechanism to map the feature maps from the CNN-Encoder to a predefined number of object slots and has been inspired by the transformer models from the NLP world.

We investigate the utility of this architecture when used together with realistic video footage. 
Therefore, we **implemented the SetCon with pytorch** according to its description and build upon it to meet our requirements.
We then created two different datasets, in which we film given objects from different angles and distances, similar to Pirk [[2]](#2) ([Github](https://online-objects.github.io/), [Paper](https://arxiv.org/abs/1906.04312)). However, they relied on a faster-RCNN for the object detection, whereas we are keen to extract the objects solely by leveraging the contrastive loss and the slot attention module.
By training a decoder on top of the learned representations, we found that in many cases the model can successfully extract objects from a scene.

This repo contains our pytorch-implementation of the SetCon according to the authors description. **Note, this is not the official implementation.** If you have questions, feel free to reach out to me. 


## Results
For our work, we have taken two videos, a Three-Object video and a Seven-Object video. In these videos we interacted with the given objects and moved them to different places and constantly changed the view perspective. Both are 30mins long, such that each contains about 54.000 frames.


<p align="center">
<img width="850" alt="eval_3_obj" src="https://user-images.githubusercontent.com/44442845/129076811-0f4d081f-d76d-498a-aa09-6534cdd91553.png"><br/>
 Figure 1: An example of the object extraction on the test set of the Three-Object dataset.
</p>

We trained the contrastive pretext model (SetCon) on the first 80% and then evaluated the learned representations on the remaining 20%.
Therefore, we trained a decoder, similar to the evaluation within the SetCon paper and looked into the specialisation of each slot. Figures 1 and 2 display two evaluation examples, from the test-set of the Three-Object Dataset and the Seven-Object Dataset. Bot figures start with the ground truth for three time stamps. During evaluation only the ground truth at t will be used to obtain the reconstructed object slots as well as their alpha masks.
The Seven-Object video is itended to be more complex and one can perceive in figure 2 that the model struggles more than on the Three-Obejct dataset to route the objects to slots. On the Three-Object dataset, we achieved 0.0043 ± 0.0029 MSE and on the Seven-Object dataset 0.0154 ± 0.0043 MSE.




<p align="center">
<img width="850" alt="eval_7_obj" src="https://user-images.githubusercontent.com/44442845/129076832-c50baff0-7c30-41d8-b5c4-05328c0126b8.png"><br/>
  Figure 2: An example of the object extraction on the test set of the Seven-Object dataset.
</p>


## How to use

For our work, we have taken two videos, a Three-Object video and Seven-Object video. Both datasets are saved as frames and are then encoded in a h5-files.
To use a different dataset, we further provide a python routine `process frames.py`, which converts frames to h5 files.

For the contrastive pretext-task, the training can be started by:
```
python3 train_pretext.py --end 300000 --num-slots 7
        --name pretext_model_1 --batch-size 512
        --hidden-dim=1024 --learning-rate 1e-5
        --feature-dim 512 --data-path ’path/to/h5file’
```
        
Further arguments, like the size of the encoder or for an augmentation pipeline, use the flag `-h` for help.
Afterwards, we freezed the weights from the encoder and the slot-attention-module and trained a downstream decoder on top of it.
The following command will train the decoder upon the checkpoint file from the pretext task:

```
python3 train_decoder.py --end 250000 --num-slots 7
        --name downstream_model_1 --batch-size 64
        --hidden-dim=1024 --feature-dim 512
        --data-path ’path/to/h5file’
        --pretext-path "path/to/pretext.pth.tar"
        --learning-rate 1e-5
```        
        
For MSE evaluation on the test-set, use both checkpoints, from the pretext- model for the encoder- and slot-attention-weights and from the downstream- model for the decoder-weights and run:
```
python3 eval.py --num-slots 7 --name evaluation_1
        --batch-size 64 --hidden-dim=1024
        --feature-dim 512 --data-path ’path/to/h5file’
        --pretext-path "path/to/pretext.pth.tar"
        --decoder-path "path/to/decoder.pth.tar"
```

## Implementation Adjustments

Insead of many small sequences of articially created frames, we need to deal with a long video-sequence. Therefore, each element in our batch mirrows a single frame at a given time t, not a sequence. For this single frame at time t, we load its two predecessors, which are then used to predict the frame at t, and thereby create a positive example.
Further, we found, that the infoNCE-loss to be numerically unstable in our case, hence we opted for the almost identical but more stable NT-Xent in our implementation.



## References

<a id="1">[1]</a> 
Löwe, Sindy et al. (2020). 
Learning object-centric video models by contrasting sets.
Google Brain team.

<a id="2">[2]</a> 
Pirk, Sören et al. (2019). 
Online object representations with contrastive learning. 
Google Brain team.

<a id="3">[3]</a> 
Locatello, Francesco et al. 
Object-centric learning with slot attention. 



