# Fairer-AI

This is the official code of the paper:

# Fairer AI in Ophthalmology via Implicit Fairness Learning for Mitigating Sexism and Ageism (submitted to Nature Commnunications)
## Abstract
The transformative impact of artificial intelligence (AI) in various domains indicates that it should be not only accurate but also objective and fair. Unfair AI systems in ophthalmology pose enormous challenges that may impact the delivery of fair and unbiased healthcare. Here, we propose a fairer AI model (called FairerOPTH) to mitigate AI sexism and ageism due to the inherent biases based on differing the incidence of eye diseases across ages and sex (biological attribute). Specifically, we incorporate the causal relationship between fundus features and disease diagnosis into AI model training, where the causal relationship is invariant to sensitive attributes such as race, sex and age. To evaluate this implicit fairness learning method based on the causal relationship between fundus features and diagnosis, we additionally collect the largest and most diverse fundus image dataset with data from over 8,405 patients representing a wide age range (0 to 90 years). The fundus dataset contains two types of advanced ultra-widefield and regular narrow-angle fundus images, with the ultra-widefield imaging dataset containing 16,530 fundus images annotated with 38 ophthalmic diseases and 67 fundus features and the narrow-angle imaging dataset containing 4,540 fundus images annotated with 16 ophthalmic diseases and 20 fundus features. Through extensive evaluation and comparison with state-of-the-art approaches, we demonstrate the significant ability of FairerOPTH to mitigate sexism and ageism. Using Shannon entropy theory, we also mathematically prove that incorporating the causal relationship between fundus features and diagnosis can indeed improve the performance of the fundus disease identification model. Our results highlight the potential of this implicit fairness learning method to promote fair and inclusive ophthalmological care, ensuring equitable treatment for patients regardless of their sex or age.

<img title="Details of ultra-widefield and narrow-angle datasets" src="https://github.com/mintanwei/Fairer-AI/blob/main/A1_dataset_intro.tif" width=60%>

## Codes and Models:
### Requirement:
PyTorch 1.10.1, torchvision 0.11.2. The code is tested with python=3.7, cuda=11.4.

### Prepare:
1. Download MixNAF (narrow-angle dataset) and OcluloScope ( ultra-widefield dataset) at https://drive.google.com/drive/folders/1XUaillgNC0Xx60fdxMn-FfffFMNMU-jg?usp=drive_link.
2. Update the "MixNAF_dir" and "OculoScope_dir" in config.py.

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

![Activation maps of FairerOPTH](https://github.com/mintanwei/Fairer-AI/blob/main/CAMs.jpg)

### Contact
Email: wmtan@fudan.edu.cn
