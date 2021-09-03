# Attack as Defense

This repository contains the code for the ISSTA'21 paper [Attack as Defense: Characterizing Adversarial Examples using Robustness](https://dl.acm.org/doi/10.1145/3460319.3464822).

[![Language grade: Python](https://img.shields.io/lgtm/grade/python/g/persistz/attack-as-defense.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/persistz/attack-as-defense/context:python)

## Requirements & Setup

The experiment environment is based on the baseline, https://github.com/rfeinman/detecting-adversarial-samples ("Detecting Adversarial Samples from Artifacts", Feinman et al. 2017).

We implemented 'Attack as Defense' based on this repo.

Req:
> Foolbox == 1.8.0


## Code Structure

/attack_method, attack methods code, from [Foolbox](https://github.com/bethgelab/foolbox).

/data, samples and models

/results, output results

/scripts, scripts that users will run


## Running the Code

Follow the baseline, all of the scripts for running the various parts of the code are located in the scripts/ subfolder.


In this repo, we will show how to run 'attack as defense' method on the MNIST dataset.

We provide a natural trained model in the data folder so user do not need to train a new model, the training steps are followed the baseline repo. We also provide the adversarial examples generate by the baseline, for C&W attack, as the baseline has not provide this part of the attack code yet, so we used the Foolbox tool with default parameters to generate these examples.
  

### 1. Attack adversarial examples

`CUDA_VISIBLE_DEVICES=XX python attack_iter_generate.py -a BIM -i fgsm`

where `<-a>` defines the attack methods used for defense,
and `<i>` defines the input data.

After executing this code, we can generate the BIM attack costs of fgsm examples.
 
User can use the scripts `bash attack_iter_generate.sh` to generate adversarial and benign samples attack costs automatically.
 
### 2. Attack benign samples 

`CUDA_VISIBLE_DEVICES=XX python attack_iter_generate.py -a BIM -i benign`

User can use the scripts `bash attack_iter_generate.sh` to generate adversarial and benign samples attack costs automatically.

### 3. Get the AUROC results

`python calc_auc.py -a BIM`

where `<-a>` defines the attack methods used for defense.

### 4. Detect

The code contains two detection methods, k-nn based detector and z-score based detector.

`CUDA_VISIBLE_DEVICES=XX python attack_as_defense_detector.py -d knn -a BIM --init
`

`CUDA_VISIBLE_DEVICES=XX python attack_as_defense_detector.py -d zscore -a BIM
`

For the first time to execute it, please add the `--init` parameter, 
which generates the training set for the detection.


