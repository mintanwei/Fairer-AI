# Fairer AI

This is the official code of the paper:

# Fairer AI in Ophthalmology via Implicit Fairness Learning for Mitigating Sexism and Ageism

<img title="Details of ultra-widefield and narrow-angle datasets" src="https://github.com/mintanwei/Fairer-AI/blob/main/dataset_intro.jpg" width="80%">

## Codes and Models:
### Requirement:
PyTorch 1.10.1, torchvision 0.11.2. The code is tested with python=3.7, cuda=11.4.

### Train:
#### for ultra-widefield
Run OculoScope_train.py to perform training. Checkpoint will be saved to ./saved_models/OculoScope/.
#### for narrow-angle
Run MixNAF_train.py to perform training. Checkpoint will be saved to ./saved_models/MixNAF/.

### Test:
#### for ultra-widefield
Run OculoScope_val.py to get the results of the accuracy experiment.
Run OculoScope_fairness_analysis.py to get the results of age and gender fairness experiments.
#### for narrow-angle
Run MixNAF_val.py to get the results of the accuracy experiment.

<img title="Activation maps of FairerOPTH" src="https://github.com/mintanwei/Fairer-AI/blob/main/CAMs.jpg" width="60%">
