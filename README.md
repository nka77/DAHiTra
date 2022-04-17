# Remote Sensing Damage Classification and Change Detection with Transformers

## Requirements
```
Python 3.6
pytorch 1.6.0
torchvision 0.7.0
einops  0.3.0
```

## Installation

Clone this repo:
```shell
git clone https://github.com/nka77/DamageAssessment.git
cd DamageAssessment
```

## Train
Please refer the training script `run_cd.sh` and the evaluation script `eval.sh` in the folder `scripts`. 


## Dataset Preparation

### Data structure

```
"""
xBD damage classification data set with pixel-level binary labels；
├─train
    |-images
    ├─masks
├─tier3
    |-images
    ├─masks
├─test
    |-images
"""
```

`train` and `tier3` : pre-disaster and post-disaster images;
`masks`: 5 class label maps;
```
"""
LEVIR Change detection data set with pixel-level binary labels；
├─A
├─B
├─label
└─list
"""
```

`A`: images of t1 phase;
`B`:images of t2 phase;
`label`: label maps;
`list`: contains `train.txt, val.txt and test.txt`, each file records the image names (XXX.png) in the change detection dataset.

### Data Download 

xBD: https://xview2.org/dataset
LEVIR-CD: https://justchenhao.github.io/LEVIR/

## Acknowledgements

We thank https://github.com/justchenhao/BIT_CD.git for providing the code base publicly. We develop our code on the top of this repository.