## Learning Fast Adaptation on Cross-Accented Speech Recognition
### Genta Indra Winata, Samuel Cahyawijaya, Zihan Liu, Zhaojiang Lin, Andrea Madotto, Peng Xu, Pascale Fung

<img src="img/pytorch-logo-dark.png" width="10%"> [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) 

This is the implementation of our paper accepted in Interspeech 2020 and the pre-print can be downloaded [here] (https://arxiv.org/pdf/2003.01901.pdf).

This code has been written using PyTorch. If you use any source codes or datasets included in this toolkit in your work, please cite the following paper.
```
@article{winata2020learning,
  title={Learning fast adaptation on cross-accented speech recognition},
  author={Winata, Genta Indra and Cahyawijaya, Samuel and Liu, Zihan and Lin, Zhaojiang and Madotto, Andrea and Xu, Peng and Fung, Pascale},
  journal={arXiv preprint arXiv:2003.01901},
  year={2020}
}
```

## Abstract
Local dialects influence people to pronounce words of the same language differently from each other. The great variability and complex characteristics of accents creates a major challenge for training a robust and accent-agnostic automatic speech recognition (ASR) system. In this paper, we introduce a cross-accented English speech recognition task as a benchmark for measuring the ability of the model to adapt to unseen accents using the existing CommonVoice corpus. We also propose an accent-agnostic approach that extends the model-agnostic meta-learning (MAML) algorithm for fast adaptation to unseen accents. Our approach significantly outperforms joint training in both zero-shot, few-shot, and all-shot in the mixed-region and cross-region settings in terms of word error rate.

## Data

## Model Architecture

## Setup
- Install PyTorch (Tested in PyTorch 1.0 and Python 3.6)
- Install library dependencies (requirement.txt)

## Run the code
- MAML
```

```

- Joint training
```

```

## Bug Report
Feel free to create an issue or send email to giwinata@connect.ust.hk or scahyawijaya@connect.ust.hk
