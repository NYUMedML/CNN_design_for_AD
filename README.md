# On the design of convolutional neural networks for automatic detection of Alzheimer’s disease

This repository contains the code for the paper [On the design of convolutional neural networks for automatic detection of Alzheimer’s disease](). 
## ADNI dataset
Here are some examples of scans for each categories in the dataset:
<img src="data_examples/CN_example.png" width="600" /> 
<img src="data_examples/MCI_example.png" width="600" /> 
<img src="data_examples/AD_example.png" width="600" /> 
## Requirements
- Python 3.6
- PyTorch 0.4
- torchvision
- progress
- matplotlib
- numpy
- visdom

## Data Preprocessing
Data Preprocessing with Clinica:
1. To convert data into BIDS format, please read the docs on Clinica [website](http://www.clinica.run), install required softwares and download the required clinical files. You can find the script in /datasets/files: 
```
run_convert.sh
```

2. To preprocess converted splitted data, use the files in /datasets/files. For training data, refer:
```
run_adni_preprocess.sh
```

## Neural Network Training
Train the network ADNI dataset:

```
python main.py
```

You can create your own config files and add a **--config** flag to indicate the name of your config files.

The trained best model (with widening factor 8 and adding age) can be found [here](https://drive.google.com/file/d/1zU21Kin9kXg_qmj7w_u5dGOjXf1D5fa7/view?usp=sharing). We also provide the [link](https://drive.google.com/file/d/1KurgyjQP-KReEO0gf31xxjwE5R-xuSRB/view?usp=sharing) to download the template we used for preprocessing. 


## Results
| Method             | Acc.        | Balanced Acc. | Micro-AUC  | Macro-AUC |
| ----------------- | ----------- | ----------- | -----------  | ----------- | 
| ResNet-18 3D    | 52.4%      | 53.1%           | -           | -           |
| AlexNet 3D      | 57.2%      | 56.2%           | 75.1%       | 74.2%       |
| X 1             | 56.4%      | 54.8%           | 74.2%       | 75.6%       |
| X 2             | 58.4%      | 57.8%           | 77.2%       | 76.6%       |
| X 4             | 63.2%      | 63.3%           | 80.5%       | 77.0%       |
| X 8             | 66.9%      | 67.9%           | 82.0%       | 78.5%       |
| X 8 + age       | 68.2%      | 70.0%           | 82.0%       | 80.0%       |


## References
- S. Liu, C. Yadav, C. Fernandez-Granda, N. Razavian. "On the design of convolutional neural networks for automatic detection of Alzheimer’s disease", in NeurIPS ML4H, 2019.
