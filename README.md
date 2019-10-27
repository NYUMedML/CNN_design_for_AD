# On the design of convolutional neural networks for automatic detection of Alzheimer’s disease

This repository contains the code for the paper [On the design of convolutional neural networks for automatic detection of Alzheimer’s disease](). 
## ADNI data
Here are few examples of scans in the dataset:
<img src="data_examples/CN_example.png" width="800" /> 
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



## References
- S. Liu, C. Yadav, C. Fernandez-Granda, N. Razavian. "On the design of convolutional neural networks for automatic detection of Alzheimer’s disease", in NeurIPS ML4H, 2019.
